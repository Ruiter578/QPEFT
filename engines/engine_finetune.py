import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from utils import misc
import random
import time
from utils.logger import print_log
import datetime
import builtins

# flops count
from fvcore.nn import FlopCountAnalysis

from .base import EngineBase
import json
from .utils import adjust_learning_rate, calculate_hardness
from configs.params_groups import TUNE_QUANT_PARAMS

from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
import os

import numpy as np
from scipy.stats import median_abs_deviation

# class HybridClip:
class GradCliper:
    def __init__(self, model, alpha=0.5, beta=0.2, min_weight=0.1, max_weight=1., device='cuda'):
        self.trainable_params = dict()
        for n, v in model.named_parameters():
            if v.requires_grad:
                self.trainable_params.update({n: v})
        self.model = model
        self.alpha = alpha  # 范数控制系数
        self.beta = beta    # 量化感知系数
        self.ema_norm = {}  # 指数移动平均范数
        self.min_weight = min_weight

        self.scales = torch.linspace(min_weight, max_weight, len(model.blocks)).to(device)
        
    def clip(self):
        for name, param in self.trainable_params.items():
            if param.grad is not None:
                # Check params depth
                if 'patch_embed.' in name:
                    depth = 0
                elif name.startswith('blocks.'):
                    depth = int(name.split('.')[1])
                else:
                    continue    # parameters in head
                # 计算混合阈值
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                
                # 更新EMA范数
                if name not in self.ema_norm:
                    self.ema_norm[name] = grad_norm
                else:
                    self.ema_norm[name] = (1 - self.alpha) * self.ema_norm[name] + \
                                           self.alpha * grad_norm
                
                # 量化感知项
                quant_term = self.beta * param_norm
                
                # 最终阈值：EMA + 量化感知
                threshold = self.ema_norm[name] + quant_term
                
                # 分层调整
                # TODO: is directly clamp a properly choice?
                if name.endswith(TUNE_QUANT_PARAMS):    # we observe that the q params are disproper
                    threshold *= self.scales[depth]
                # param.grad.data.clamp_(-threshold, threshold)
                param.grad.data = torch.clamp(param.grad.data.clone() * self.scales[depth], -threshold, threshold)

class GradMonitor:
    def __init__(self,
                 writer,
                 trace_params=dict(block_id=[0, 5, 11],
                                   param_ends=['qkv.lora_B', 'mlp.fc2.alpha', 'mlp.fc2.lora_B', 
                                               'attn_quantizer.log_base_alpha',
                                               'attn_quantizer.adaptive_shift', 
                                               ])):
        self.writer = writer
        self.trace_params = trace_params
        self.inited = False

    def _filter(self, trainable_params_dict):
        keys = list(trainable_params_dict.keys())
        monitor_dict = dict()
        for k in keys:
            if k.endswith(tuple(self.trace_params['param_ends'])):
                if k.startswith('blocks'):
                    block_id = int(k.split('.')[1])
                    if block_id in self.trace_params['block_id']:
                        monitor_dict.update({k: trainable_params_dict[k]})
                else:
                    continue
        self.monitor_dict = monitor_dict
        self.inited = True
    def __call__(self, trainable_params_dict, glob_step, prefix='Grad'):
        if not self.inited:
            self._filter(trainable_params_dict)
        log_dict = defaultdict(dict)
        for k, v in self.monitor_dict.items():
            g_norm = v.grad.data.norm().item()
            for ends in self.trace_params['param_ends']:
                if k.endswith(ends):
                    p_type = ends
                    break
            block_id = int(k.split('.')[1])
            log_dict[p_type].update({f"block_{block_id}": g_norm})

        for k, v in log_dict.items():
            self.writer.add_scalars(f"{prefix}/{k}", v, glob_step)

class FineTuneEngine(EngineBase):
    def __init__(self, logger, args, device='cuda'):
        self.device = device
        super().__init__(logger, args)

    def train(self,
              model: torch.nn.Module,
              criterion: torch.nn.Module,
              train_loader: Iterable,
              val_loader: Iterable,
              optimizer: torch.optim.Optimizer,
              max_norm: float = 0,
              model_ema: Optional[ModelEma] = None,
              mixup_fn: Optional[Mixup] = None,
              amp: bool = False,
              writer=None
              ):
        self.writer = writer
        sq_log_dir=f"{self.args.log_path}/exp_avg_sq"
        os.mkdir(sq_log_dir)
        self.sq_writer = SummaryWriter(log_dir=sq_log_dir)
        self.amp_scaler = torch.cuda.amp.GradScaler() if amp else None
        self.logger.info("Start training...")
        self.glob_train_step = 0
        self.trainable_params = dict()
        for n, v in model.named_parameters():
            if v.requires_grad:
                self.trainable_params.update({n: v})
                
        start_time = time.time()
        max_accuracy = 0.0
        approx_max_accuracy = 0.0
    
        # self.grad_cliper = GradCliper(model, alpha=0.5, beta=0.2, min_weight=0.1, max_weight=1., device=self.device)
        self.grad_monitor = GradMonitor(writer=writer)
        self.dry_run = True # for init some params
        self.adjust_qdrop = False
        if getattr(self.args, 'rounding', False):
            if getattr(self.args.rounding, 'strategy', None) == 'drop_ste':
                if self.args.rounding.drop_epc_ratio == 0:
                    pass
                elif 0 < self.args.rounding.drop_epc_ratio <= 1:
                    self.adjust_qdrop = getattr(self.args.rounding, 'adjust_from_grads', False)
                else:
                    raise ValueError("drop_epc_ratio must be in [0, 1]")

        if self.adjust_qdrop:
            self.module_params_adjust_qdrop, self.modules_adjust_qdrop = find_rounding_learnable_modules(model, optimizer)

        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.args.epochs):
            train_stats = self.train_one_epoch(
                    model=model,
                    criterion=criterion,
                    train_loader=train_loader,
                    epoch=epoch,
                    optimizer=optimizer,
                    max_norm=max_norm,
                    model_ema=model_ema,
                    mixup_fn=mixup_fn,
                    amp=amp,
                )

            if epoch > 0 and (epoch % self.args.val_interval == 0 or epoch == self.args.epochs-1):
                val_stats = self.evaluate(
                        model=model,
                        val_loader=val_loader,
                        criterion=criterion
                    )

                self.logger.info(f"Accuracy of the network on the {len(val_loader.dataset)} test images: {val_stats['acc1']:.1f}%")
                is_eval_approx_rounding = False
                if getattr(self.args, 'rounding', None) is not None:
                    if self.args.rounding.strategy == 'tanh_progressive':
                        approx_epochs = self.args.rounding.num_epochs
                        if epoch <= approx_epochs:
                            is_eval_approx_rounding = True

                if is_eval_approx_rounding:
                    prev_max_accuracy = approx_max_accuracy
                    approx_max_accuracy = max(approx_max_accuracy, val_stats["acc1"])
                    self.logger.info(f'Max approx accuracy: {approx_max_accuracy:.2f}%')
                else:
                    prev_max_accuracy = max_accuracy
                    max_accuracy = max(max_accuracy, val_stats["acc1"])
                    self.logger.info(f'Max accuracy: {max_accuracy:.2f}%')

                log_stats = {'eval_approx_rounding': is_eval_approx_rounding, **{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch}
                for k, v in val_stats.items():
                    self.writer.add_scalar(f"Val/epoch_{k}", v, epoch)

                with open(f"{self.args.log_path}/log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if not is_eval_approx_rounding:
                    if self.args.log_path and misc.is_main_process():
                        if self.args.save_ckpt and max_accuracy > prev_max_accuracy:
                            checkpoint_paths = [f"{self.args.log_path}/best_checkpoint.pth"]
                            for checkpoint_path in checkpoint_paths:
                                torch.save(
                                        dict(
                                            model=model.state_dict(),
                                            optimizer=optimizer,
                                            epoch=epoch,
                                            scaler=self.amp_scaler.state_dict() if amp else None,
                                            args=self.args
                                            ),
                                        checkpoint_path
                                    )
            if epoch > 0 and epoch % getattr(self.args, 'epochs_reset_qparams', 10000) == 0:
                if self.adjust_qdrop:
                    self.set_dropout_from_grads(model, optimizer, epoch)
                    self.logger.info(f"Dropout rates have been adjusted!")
                has_been_reset = False
                if self.args.epochs - epoch < 10:
                    pass    # do not reset qparams for the last 10 epochs
                elif hasattr(model, '_reset_qparams'):
                    model._reset_qparams()
                    has_been_reset = True
                elif hasattr(model, 'module'):
                    assert hasattr(model.module, '_reset_qparams')
                    model.module._reset_qparams()
                    has_been_reset = True
                else:
                    raise NotImplementedError(f"Model {model.__class__.__name__} has no _reset_qparams method!")
                if has_been_reset:
                    self.logger.info(f"Quant parameters for weights have been reset!")
            is_approx_rounding = False
            hardness = None
            if getattr(self.args, 'rounding', None) is not None:
                if self.args.rounding.strategy == 'tanh_progressive':
                    if epoch < approx_epochs:
                        hardness = calculate_hardness(self.args, epoch)
                        if hasattr(model,'_reset_approx_hardness'):
                            model._reset_approx_hardness(hardness)
                        elif hasattr(model,'module'):
                            assert hasattr(model.module,'_reset_approx_hardness')
                            model.module._reset_approx_hardness(hardness)
                        else:
                            raise NotImplementedError(f"Model {model.__class__.__name__} has no _reset_approx_hardness method!")
                        is_approx_rounding = True
                    elif epoch == approx_epochs:
                        if hasattr(model, '_reset_rounding_strategy'):
                            model._reset_rounding_strategy(None)    # None for ste
                        elif hasattr(model, 'module'):
                            assert hasattr(model.module, '_reset_rounding_strategy')
                            model.module._reset_rounding_strategy(None)
                        else:
                            raise NotImplementedError(f"Model {model.__class__.__name__} has no _reset_rounding_strategy method!")
                        is_approx_rounding = True
                    else:
                        pass
            self.logger.info(f"train_cfgs={self.args.train_cfgs}, quant_cfgs={self.args.quant_cfgs}, lora_cfgs={self.args.lora_cfgs}. During approx rounding={is_approx_rounding}, hardness={hardness}")
        if self.args.log_path and misc.is_main_process():
            with open(f"{self.args.log_path}/log.txt", "a") as f:
                f.write(f"max acc = {max_accuracy:.2f}%")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('Training time {}'.format(total_time_str))

        return max_accuracy
    
    def train_one_epoch(self,
                        model: torch.nn.Module,
                        criterion: torch.nn.Module,
                        train_loader: Iterable,
                        epoch: int,
                        optimizer: torch.optim.Optimizer,
                        max_norm: float = 0,
                        model_ema: Optional[ModelEma] = None,
                        mixup_fn: Optional[Mixup] = None,
                        amp: bool = True,):
            model.train()
            criterion.train()

            metric_logger = misc.MetricLogger(delimiter="  ", logger=self.logger)
            # metric_logger.add_meter('lr_g0', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = 'Epoch: [{}]'.format(epoch)
            print_freq = 10

            # for samples, targets in metric_logger.log_every(train_loader, print_freq, header):
            for data_iter_step, batch in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
                samples = batch[0].to(self.device, non_blocking=True)
                targets = batch[1].to(self.device, non_blocking=True)
                adjust_learning_rate(optimizer=optimizer, epoch=data_iter_step / len(train_loader) + epoch, args=self.args)
                if mixup_fn is not None:
                    samples, targets = mixup_fn(samples, targets)

                if self.dry_run:
                    # with torch.cuda.amp.autocast(enabled=amp):
                    #     outputs = model(samples)
                    #     model.zero_grad()
                    with torch.no_grad():
                        model.eval()
                        _ = model(samples)
                    model.train()
                    self.dry_run = False
                        
                with torch.cuda.amp.autocast(enabled=amp):
                    outputs = model(samples)
                    loss = criterion(outputs, targets)

                loss_value = loss.item()

                grad_clip = False
                if not math.isfinite(loss_value):
                    self.logger.info("Loss is {}, clipping gradient".format(loss_value))
                    grad_clip = True

                optimizer.zero_grad()
                # this attribute is added by timm on one optimizer (adahessian)
                if amp:
                    # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                    # loss_scaler(loss, optimizer, clip_grad=max_norm,
                    #         parameters=model.parameters(), create_graph=is_second_order)
                    # raise NotImplementedError
                    self.amp_scaler.scale(loss).backward()
                    self.amp_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place

                else:
                    loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.trainable_params(), 10)

                self.grad_monitor(self.trainable_params, self.glob_train_step, prefix='Grad')

                if amp:
                    self.amp_scaler.step(optimizer)
                    self.amp_scaler.update()
                else:
                    optimizer.step()

                torch.cuda.synchronize()
                if model_ema is not None:
                    model_ema.update(model)

                metric_logger.update(loss=loss_value)
                cur_lr = optimizer.param_groups[0]["lr"]
                metric_logger.update(lr=cur_lr)

                self.writer.add_scalar("Train/iter_loss", loss_value, self.glob_train_step)
                self.writer.add_scalar("Train/iter_lr", cur_lr, self.glob_train_step)

                
                self.glob_train_step += 1
                # metric_logger.update(lr_g1=optimizer.param_groups[1]["lr"])
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            # self.logger.info("Averaged stats:", metric_logger.meters)
            return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def set_dropout_from_grads(self, model, optimizer, cur_epoch):
        # STD_WEIGHT = 0.25

        DROP_EPC_RATIO = getattr(self.args.rounding,'drop_epc_ratio', 0.9)  # 0.9 > 0.8
        
        progress = min(1., cur_epoch / math.floor(self.args.epochs * DROP_EPC_RATIO))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        # outlier_base_prob = self.args.rounding.outliers_drop_rate * cosine_decay
        # self.logger.info(f"Epoch {cur_epoch}, outlier_base_prob: {outlier_base_prob}")
        param_states = process_modules_grad_stats(self.module_params_adjust_qdrop, optimizer)
        difficulties = {}
        # state_dict = model.state_dict()
        for m_name, states in param_states.items():
            state_mean, state_std = states
            # difficulties[m_name] = state_mean + STD_WEIGHT * state_std
            difficulties[m_name] = state_mean
            self.sq_writer.add_scalar(f"{m_name}", state_mean, cur_epoch)

        # total_difficulty = sum(difficulties.values())
        # diff_values = torch.tensor(list(difficulties.values()))
        diff_values = np.array(list(difficulties.values()))
        # if MODE == 'z_score':
            # sens_map = sensitivity_mapping(diff_values)
        zscores_ = robust_zscore(diff_values)
        sensitivity_scores = {name: zscores_[i] for i, name in enumerate(difficulties.keys())}

        cur_min_prob = 10.
        cur_max_prob = -1.
        set_probs = []
        z_threshold = 100.

        round_cfg = self.args.rounding
        outlier_prob = (round_cfg.outliers_drop_rate - round_cfg.final_drop_rate) * cosine_decay + round_cfg.final_drop_rate
        inlier_prob = (round_cfg.initial_drop_rate - round_cfg.final_drop_rate) * cosine_decay + round_cfg.final_drop_rate
        outlier_info = []
        for m_name, score in sensitivity_scores.items():
            # target_module = getattr(model, m_name)
            target_module = self.modules_adjust_qdrop[m_name]
            assert hasattr(target_module, 'rounding')
            assert hasattr(target_module.rounding, 'reset_drop_prob')

            if score >= z_threshold:
                prob  = outlier_prob
                outlier_info.append((m_name, score))
            else:
                prob = inlier_prob
            
            m_prob = max(0.0, min(1.0, prob))
            
            if m_prob < cur_min_prob:
                cur_min_prob = m_prob
            if m_prob > cur_max_prob:
                cur_max_prob = m_prob

            target_module.rounding.reset_drop_prob(m_prob)
            set_probs.append(m_prob)
            # self.logger.info(f"module: {m_name}, z_score: {score:.4f}, set_prob: {m_prob:.4f}")
        self.logger.info(f"outlier_info: {outlier_info}")
        self.logger.info(f"outlier_prob: {outlier_prob:.4f}, inlier_prob: {inlier_prob:.4f}")
        self.logger.info(f"number of outliers: {len(outlier_info)}, number of inliers: {len(sensitivity_scores) - len(outlier_info)}")
        self.logger.info(f"Epoch {cur_epoch}, min_prob: {cur_min_prob:.4f}, max_prob: {cur_max_prob:.4f}")
        self.writer.add_scalar("Train/min_prob", cur_min_prob, cur_epoch)
        self.writer.add_scalar("Train/max_prob", cur_max_prob, cur_epoch)
        # self.writer.add_scalar("Train/base_prob", outlier_base_prob, cur_epoch)
        self.writer.add_scalar("Train/mean_prob", np.mean(set_probs), cur_epoch)
    @torch.no_grad()
    def evaluate(self, model, val_loader, criterion):
        metric_logger = misc.MetricLogger(delimiter="  ", logger=self.logger)
        header = 'Test:'
        # switch to evaluation mode
        model.eval()

        for images, target in metric_logger.log_every(val_loader, 10, header):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            # compute output

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp=True):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_video(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None,
                    deit=False, args=None):

    model.train()
    criterion.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10


    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets = batch[0], batch[1]
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # we use a per iteration (instead of per epoch) lr scheduler
        # if data_iter_step % accum_iter == 0:
        #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)

        else:

            if not deit:
                outputs = model(samples)
            else:
                outputs, _ = model(samples)

            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        grad_clip = False
        if not math.isfinite(loss_value):
            print("Loss is {}, clipping gradient".format(loss_value))
            grad_clip = True

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.trainable_params(), 10)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# def find_rounding_learnable_modules(model) -> list[str]:
#     target_module_name = []
#     for name, module in model.named_modules():
#         if hasattr(module, 'rounding'):
#             # print(f"Module: {name}, Rounding: {module.rounding}")
#             is_target_module = False
#             for p in module.parameters():
#                 if p.requires_grad:
#                     is_target_module = True
#                     break
#             if is_target_module:
#                 target_module_name.append(name)
#     return target_module_name
@torch.no_grad()
def find_rounding_learnable_modules(model, optimizer):
    target_modules = dict()
    target_module_params = defaultdict(list)
    for name, module in model.named_modules():
        is_target_module = False
        if hasattr(module, 'rounding'):
            if not hasattr(module.rounding, 'reset_drop_prob'):
                continue            
            # print(f"Module: {name}, Rounding: {module.rounding}")
            for n, p in module.named_parameters(recurse=False): # do not searching sub-modules
                if p.requires_grad:
                    p_in_optim = False
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            if param is p:
                                p_in_optim = True
                                break
                        if p_in_optim:
                            break
                    if p_in_optim:
                        target_module_params[name].append(p)
                        is_target_module = True
                    else:
                        raise ValueError(f"Learnable parameter {n} in module {name} is not in optimizer!")
            if is_target_module:
                target_modules[name] = module
    return target_module_params, target_modules

@torch.no_grad()
def process_modules_grad_stats(target_module_params: dict, optimizer: torch.optim.Optimizer):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Only support AdamW optimizer for now!")
    param_states = dict()
    for name, params in target_module_params.items():
        cur_stats = []
        for p in params:
            assert p.requires_grad
            state = optimizer.state[p]  # dict_keys(['step', 'exp_avg', 'exp_avg_sq'])
            if 'exp_avg_sq' not in state:
                raise ValueError(f"Parameter in module {name} has no exp_avg_sq in optimizer state!")
            exp_avg_sq = state['exp_avg_sq'].flatten().cpu().numpy()
            cur_stats.append(exp_avg_sq)
        # cur_stats = torch.cat(cur_stats)
        cur_stats = np.concatenate(cur_stats)
        # param_states[name] = (cur_stats.mean(), cur_stats.std())
        param_states[name] = (cur_stats.mean(), None)

    return param_states
            

@torch.no_grad()
def evaluate_video(data_loader, model, device, amp=True):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()

    targets = []
    predictions = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        B, V = images.shape[0], images.shape[1]
        images = images.flatten(0, 1)
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                output = output.view(B, V, -1).mean(dim=1)
                loss = criterion(output, target)
        else:
            output = model(images)
            output = output.view(B, V, -1).mean(dim=1)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        predictions.append(output)
        targets.append(target)
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    targets = torch.cat(targets, dim=0) # targets.shape 6149
    predictions = torch.cat(predictions, dim=0)
    
    metric_logger.synchronize_between_processes()
    if misc.is_dist_avail_and_initialized():
        targets = all_gather_concat(targets)
        predictions = all_gather_concat(predictions)

    acc1, acc5 = accuracy(predictions, targets, topk=(1, 5))
    print(f'* Acc@1 {acc1:.3f} Acc@5 {acc5:.3f}')
    print('=', {k: meter.global_avg for k, meter in metric_logger.meters.items()})

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def all_gather(data):

    world_size = misc.get_world_size()
    if world_size == 1:
        return [data]

    gather_list = [
        torch.empty_like(data)
        for _ in range(world_size)
    ]

    torch.distributed.all_gather(gather_list, data)

    return gather_list



def all_gather_concat(data: torch.Tensor) -> torch.Tensor:
    """Gather tensors with different first-dimension size and concat to one
    tenosr.

    Note:
        Only the first dimension should be different.

    Args:
        data (Tensor): Tensor to be gathered.

    Returns:
        torch.Tensor: The concatenated tenosr.
    """
    if misc.get_world_size() == 1:
        return data

    data_size = torch.tensor(data.size(0), device=data.device)

    sizes_list = all_gather(data_size)

    max_length = max(sizes_list)
    size_diff = max_length.item() - data_size.item()
    if size_diff:
        padding = torch.zeros(
            size_diff, *data.size()[1:], device=data.device, dtype=data.dtype)
        data = torch.cat((data, padding))

    gather_list = all_gather(data)

    all_data = []
    for tensor, size in zip(gather_list, sizes_list):

        all_data.append(tensor[:size])

    return torch.cat(all_data)



def sensitivity_mapping(sensitivity_scores: torch.Tensor) -> torch.Tensor:
    """
    将敏感度分数优雅映射到[-1,1]区间
    参数:
        sensitivity_scores: 原始敏感度张量 (形状[n])
    返回:
        映射后的分数张量 (形状[n])
    """
    # 1. 计算经验累积分布（消除量级差异）
    sorted_s, _ = torch.sort(sensitivity_scores)
    ranks = torch.searchsorted(sorted_s, sensitivity_scores).float()
    cdf = (ranks + 0.5) / len(sensitivity_scores)  # 连续CDF估计
    
    # 2. 非线性压缩（适应偏态分布）
    z = torch.erfinv(2 * cdf - 1)  # 高斯分位数变换
    compressed = torch.tanh(z)  # 压缩到(-1,1)
    
    # 3. 动态边界校准
    q_min, q_max = torch.quantile(compressed, torch.tensor([0.01, 0.99]))
    scaled = 2 * (compressed - q_min) / (q_max - q_min + 1e-8) - 1
    
    return torch.clamp(scaled, -1, 1)

def get_drop_probs(
    sensitivity_scores: torch.Tensor, 
    p_range: tuple = (0.2, 0.8)
) -> torch.Tensor:
    """
    生成与敏感度正相关的丢弃概率
    
    参数:
        sensitivity_scores: 原始敏感度张量
        p_range: 概率边界 (min_prob, max_prob)
    返回:
        丢弃概率张量
    """
    s_star = sensitivity_mapping(sensitivity_scores)
    p_min, p_max = p_range
    return p_min + (p_max - p_min) * (s_star + 1) / 2

def adjust_drop_probs(ori_s, mapping_range):
    mapped_sensitivity = sensitivity_mapping(ori_s)
    return get_drop_probs(mapped_sensitivity, mapping_range)

def robust_zscore(data):
    median = np.median(data)
    mad = median_abs_deviation(data, scale='normal')
    return (data - median) / mad