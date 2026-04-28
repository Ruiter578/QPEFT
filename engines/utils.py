import torch
import time
import math
# flops count
from fvcore.nn import FlopCountAnalysis
from easydict import EasyDict as edict

def test_model_latency(model, device, test_batch_size=512):
    T0 = 10
    T1 = 10
    speed = 0
    model.eval()
    with torch.no_grad():
        x = torch.randn(test_batch_size, 3, 224, 224).to(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.time()
        while time.time() - start < T0:   
            model(x)
        torch.cuda.synchronize()
        print("*****Test model latency (images per second)*****")
        timing = []
        while sum(timing) < T1:
            start = time.time()
            model(x)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
        timing = torch.as_tensor(timing, dtype=torch.float32)
        speed=512/timing.mean().item()
        print("Model latency: {} imgs/s".format(speed))
    
    return speed
    # exit(0)

def count_model_flops(model, device,shape=(1,3,224,224)):
    model.eval()
    rand_input = torch.rand(shape).to(device)
    flops = FlopCountAnalysis(model,rand_input)
    print("*****Count model FLOPS (GFLOPS)*****")
    total_flops = flops.total()
    print("Total FLOPS:%d, GFLOPS:%.4f"%(total_flops, float(total_flops)/1073741824))
    print(flops.by_module_and_operator())
    return float(total_flops)/1073741824


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "group_name" in param_group:
            for group_name, unfreeze_epoch in args.unfreeze_schedule.items():
                if epoch >= unfreeze_epoch and group_name==param_group['group_name']:
                    param_group["lr"] = lr * args.unfreeze_lr_ratio[group_name]
        else:                
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
    return lr
# def adjust_learning_rate(optimizer, epoch, args):
#     """Decay the learning rate with half-cycle cosine after warmup"""
#     if epoch < args.warmup_epochs:
#         lr = args.lr * epoch / args.warmup_epochs 
#     else:
#         lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
#             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
#     for param_group in optimizer.param_groups:
#         qtuning_start_epc = getattr(args, 'qtuning_start_epc', -10)
#         sched_flag = False
#         if "quant_params" in param_group:
#             if qtuning_start_epc > 0:
#                 if epoch < qtuning_start_epc:
#                     param_group["lr"] = 0.0
#                     sched_flag = True
#                 else:
#                     assert getattr(args, 'qtuning_lr_ratio') > 0
#                     param_group["lr"] = lr * args.qtuning_lr_ratio
#                     sched_flag = True

#         if not sched_flag:                
#             if "lr_scale" in param_group:
#                 param_group["lr"] = lr * param_group["lr_scale"]
#             else:
#                 param_group["lr"] = lr
#     return lr


def calculate_hardness(cfg: edict, current_epoch: int) -> float:
    """
    Dynamically calculate the hardness value based on the given configuration and the current epoch number.

    Args:
        cfg (EasyDict): A configuration object containing scheduling parameters.
                        It should include cfg.rounding.initial_hardness,
                             cfg.rounding.final_hardness,
                             cfg.rounding.num_epochs,
                             cfg.rounding.schedule.
        current_epoch (int): The current training epoch number (starting from 0).

    Returns:
        float: The calculated current hardness value.
    """
    initial_h = cfg.rounding.initial_hardness
    final_h = cfg.rounding.final_hardness
    total_epochs = cfg.rounding.num_epochs
    schedule = cfg.rounding.schedule

    if current_epoch <= 0:
        return initial_h
    if current_epoch >= total_epochs:
        return final_h

    progress = current_epoch / total_epochs

    if schedule == 'linear':
        alpha = progress
    elif schedule == 'cosine':
        alpha = 0.5 * (1 - math.cos(progress * math.pi))
    elif schedule == 'exp':
        # exp_factor = 5.0
        exp_factor = 6.0
        alpha = (math.exp(exp_factor * progress) - 1) / (math.exp(exp_factor) - 1)
    elif schedule == 'ln':
        # ln_factor = 5.0
        ln_factor = 100.0
        alpha = math.log(1 + ln_factor * progress) / math.log(1 + ln_factor)
    else:
        raise ValueError(f"Unknown scheduling strategy: '{schedule}'. "
                         f"Available values are ['linear', 'cosine', 'exp', 'ln']")

    current_hardness = initial_h + (final_h - initial_h) * alpha
    return current_hardness