import os
import sys
import torch
from torch import nn
import numpy as np
from functools import partial
import argparse
import importlib
import timm
import copy
import time

from datasets.get_datasets import build_dataset, build_pretransformed_testset
# from utils.wrap_net import *
from qpeft_quant.wrap_net_qpeft import apply_qpeft, prepare_for_peft
from utils.test_utils import *
from utils import misc
from datetime import datetime
import logging

import models
from configs import MODELS, load_configs
from utils.load_ckpt import timm_load_checkpoint, load_quant_state_dict
from timm.models import load_checkpoint

from engines import FineTuneEngine
from utils.logger import get_root_logger

from utils.optim_factory import create_optimizer

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler
import json

from torch.utils.tensorboard import SummaryWriter

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument("--model", default="deit_small",
    #                     choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large',
    #                              'deit_tiny', 'deit_small', 'deit_base', 
    #                              'swin_tiny', 'swin_small', 'swin_base', 'swin_base_384'],
    #                     help="model")
    parser.add_argument("--model", default="vitb16",
                        choices=list(MODELS.keys()),
                        help="model")
    # parser.add_argument('--config', type=str, default="./configs/vit_config.py",
    #                     help="File path to import Config class from")
    parser.add_argument('--train_cfgs', type=str, required=True,
                        help="File path to import Config class from")
    parser.add_argument('--quant_cfgs', type=str, default=None,
                        help="File path to import Config class from")
    parser.add_argument('--lora_cfgs', type=str, default=None,
                        help="File path to import Config class from")
    # parser.add_argument('--enable_qparams_training', action='store_true')
    parser.add_argument('--dataset', default="cifar",
                        help='name of dataset')

    # parser.add_argument("--num_workers", default=8, type=int,
    #                     help="number of data loading workers (default: 8)")
    parser.add_argument("--device", default="cuda", type=str, help="device")

    parser.add_argument("--amp", default=False, action="store_true", help="amp")
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--log_root", type=str, help="path to logging", default="outputs")
    parser.add_argument("--exp_name", type=str, help="name", default=None)
    parser.add_argument("--save_ckpt", action='store_true')
    parser.add_argument("--eval_ckpt", default=None, type=str, help='path to QPEFT checkpoint')
    
    # parser.add_argument('--w_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of weights')
    # parser.add_argument('--a_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of activation')
    # parser.add_argument('--s_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of post softmax activation')
    return parser


def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_model(model, args, name):
    save_path = os.path.join(args.log_path, name+'.pth')

    logging.info(f"Saving checkpoint to {save_path}")
    torch.save(model.state_dict(), save_path)


def finish_training(model):
    for name, module in model.named_modules():
        if hasattr(module, 'mode') and hasattr(module, 'reparam_bias'):
            module.reparam_bias()

def main(args):
    cfgs = load_configs(args)
    for k, v in vars(args).items():
        assert k not in cfgs.keys(), f"{k} is repeated configured!"
        setattr(cfgs, k, v)
        
    if cfgs.eval_ckpt is None:
        cfgs.eval = False
    else:
        cfgs.eval = True
    if cfgs.eval:
        cfgs.log_path = f"{cfgs.log_root}/test/{cfgs.model}/T={cfgs.train_cfgs}_Q={cfgs.quant_cfgs}_L={cfgs.lora_cfgs}/{cfgs.dataset}/{misc.get_timestamp()}"
    else:
        cfgs.log_path = f"{cfgs.log_root}/{cfgs.model}/T={cfgs.train_cfgs}_Q={cfgs.quant_cfgs}_L={cfgs.lora_cfgs}/{cfgs.dataset}/{misc.get_timestamp()}"
    if cfgs.exp_name is not None:
        cfgs.log_path += f'_{cfgs.exp_name}'
    if not os.path.exists(cfgs.log_path):
        os.makedirs(cfgs.log_path)

    if cfgs.eval:
        logger = get_root_logger(log_file=f"{cfgs.log_path}/eval.log", name=f"{cfgs.model}")
        logger.info("Start evaluation ...")
    else:
        cfgs.tb_path = f"{cfgs.log_path}/tensorboard"
        if not os.path.exists(cfgs.tb_path):
            os.makedirs(cfgs.tb_path)
        logger = get_root_logger(log_file=f"{cfgs.log_path}/run.log", name=f"{cfgs.model}")
        logger.info("Start training ...")

    if cfgs.model in ['vitb16', 'vitl']:
        cfgs.inception = True
    else:
        cfgs.inception = False
        raise NotImplementedError("Only support vitb16 and vitl")

    for name, value in cfgs.items():
        logger.info(f"{name}: {value}")
    with open(f"{cfgs.log_path}/configs.json", "w") as json_file:
        json.dump(cfgs, json_file, indent=4)

    device = torch.device(cfgs.device)

    misc.set_random_seed(cfgs.seed, deterministic=True)

    if not cfgs.eval:
        logger.info('Building training and val datasets ...')
        train_set, cfgs.nb_classes = build_dataset(is_train=True, args=cfgs)
        val_set, _ = build_dataset(is_train=False, args=cfgs)
    else:
        logger.info('Building test dataset ...')
        test_set, cfgs.nb_classes = build_dataset(is_train=False, args=cfgs)
    logger.info('Building model ...')
    # try:
    #     model = timm.create_model(model_zoo[args.model], checkpoint_path='./checkpoints/vit_raw/{}.bin'.format(model_zoo[args.model]))
    # except:
    #     model = timm.create_model(model_zoo[args.model], pretrained=True)
    full_model = models.__dict__[MODELS[cfgs.model].full_name](
            drop_path_rate=cfgs.drop_path
        )
    model = copy.deepcopy(full_model)

    if not cfgs.eval:  # TODO, training
        ckpt_path = MODELS[cfgs.model].full_ckpt_path
        assert os.path.exists(ckpt_path)
        if 'pth' in ckpt_path:
            if "best_checkpoint" in ckpt_path:
                if cfgs.nb_classes != model.head.weight.shape[0]:
                    model.reset_classifier(cfgs.nb_classes)
                incompatible_keys = timm_load_checkpoint(model, ckpt_path, strict=True)
            else:
                incompatible_keys = timm_load_checkpoint(model, ckpt_path, strict=False)
                if cfgs.nb_classes != model.head.weight.shape[0]:
                    model.reset_classifier(cfgs.nb_classes)
            logger.info(f"Incompatible Keys:\n{incompatible_keys}")
        else:
            load_checkpoint(model, ckpt_path)
            if cfgs.nb_classes != model.head.weight.shape[0]:
                model.reset_classifier(cfgs.nb_classes)
    else:
        if cfgs.nb_classes != model.head.weight.shape[0]:
            model.reset_classifier(cfgs.nb_classes)
        # pass    # evaluate ckpt do not load pretrained full precision checkpoint
    wrap_flag = False
    if cfgs.quant_cfgs is None:
        if cfgs.lora_cfgs is None:
            pass
        else:
            wrap_flag = True
    else:
        wrap_flag = True
    if wrap_flag:
        logger.info('Wraping model for quantization and/or LoRA...')
        apply_qpeft(model, cfgs)
    else:
        logger.warning("Full precision model and training head only!")

    if cfgs.lora_cfgs is None:
        pass    # TODO: QAT PEFT or full tuning
    else:
        if not cfgs.eval:
            # wrap_lora_in_attn(model, cfgs)
            prepare_for_peft(model, logger, cfgs)

    if cfgs.eval:
        eval_ckpt = torch.load(cfgs.eval_ckpt)['model']
        model.load_state_dict(eval_ckpt, strict=True)

    # TODO: load quantization configs
    logger.info(f"Model is:\n{model}")
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    engine = FineTuneEngine(logger=logger, device=device, args=cfgs)
    
    if cfgs.eval:
        test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfgs.batch_size,
        num_workers=cfgs.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=5 if cfgs.num_workers > 0 else None,
        persistent_workers=True if cfgs.num_workers > 0 else False,
        shuffle=False
        )
        metrics = engine.evaluate(model=model,
                                  val_loader=test_loader,
                                  criterion=criterion)
        max_acc = metrics['acc1']

        log_msg = "Evaluate Results:\n"

        for k, v in metrics.items():
            log_msg += f"{k}: {v:.4f}\n"
        logger.info(log_msg)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=cfgs.batch_size,
            num_workers=cfgs.num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=5 if cfgs.num_workers > 0 else None,
            shuffle=True,
            persistent_workers=True if cfgs.num_workers > 0 else False
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=cfgs.batch_size*2,
            num_workers=cfgs.num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=5 if cfgs.num_workers > 0 else None,
            persistent_workers=True if cfgs.num_workers > 0 else False,
            shuffle=False
        )
        
        if cfgs.optimizer == 'AdamW':
            optimizer = create_optimizer(cfgs, model)
        else:
            raise ValueError

        writer = SummaryWriter(log_dir=cfgs.tb_path)
        max_acc = engine.train(model=model,
                            criterion=criterion,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            amp=cfgs.amp,
                            writer=writer
                            )

    os.system(f"mv {cfgs.log_path} {cfgs.log_path}_acc={max_acc:.2f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)