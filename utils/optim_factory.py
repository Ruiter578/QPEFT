""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from timm.optim.adabelief import AdaBelief
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lamb import Lamb
from timm.optim.lars import Lars
from timm.optim.lookahead import Lookahead
from timm.optim.madgrad import MADGRAD
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from configs import TUNE_QUANT_PARAMS, TUNE_LORA_PARAMS

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


# def pre_process(model, weight_decay=1e-5, skip_list=(), cfgs=None):
#     decay = []
#     no_decay = []
#     # add_names = []
#     qparams_group = []
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue  # frozen weights
#         # TODO: Shawn: quantization params should be considering weight decay?
#         if cfgs.enable_qparams_training and name.endswith(TUNE_QUANT_PARAMS):
#             qparams_group.append(param)
#         elif len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
#             no_decay.append(param)
#         else:
#             decay.append(param)
#         # add_names.append(name)
#     # print(add_names)
#     return [
#         {'params': qparams_group, 'weight_decay': 0., "lr": 0.0, 'quant_params': True},
#         {'params': no_decay, 'weight_decay': 0., "lr": cfgs.lr},
#         {'params': decay, 'weight_decay': weight_decay, "lr": cfgs.lr}]
def pre_process(model, weight_decay=1e-5, skip_list=(), cfgs=None):
    decay = []
    no_decay = []
    # add_names = []
    # qparams_group = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        # TODO: Shawn: quantization params should be considering weight decay?
        # if cfgs.enable_qparams_training and name.endswith(TUNE_QUANT_PARAMS):
        #     qparams_group.append(param)
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            # print(name)
            no_decay.append(param)
        else:
            decay.append(param)
        # add_names.append(name)
    # print(add_names)
    # import pdb; pdb.set_trace()
    return [
        # {'params': qparams_group, 'weight_decay': 0., "lr": 0.0, 'quant_params': True},
        {'params': no_decay, 'weight_decay': 0., "lr": cfgs.lr},
        {'params': decay, 'weight_decay': weight_decay, "lr": cfgs.lr}]
# def pre_process(model, weight_decay=1e-5, skip_list=(), cfgs=None):
#     decay = []
#     no_decay = []
#     # add_names = []
#     params_dict = {}
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue  # frozen weights
#         # TODO: Shawn: quantization params should be considering weight decay?
#         if name.endswith(TUNE_QUANT_PARAMS):
#             if 'patch_embed' in name:
#                 layer_id = 0
#             else:
#                 layer_id = int(name.split(".")[1])
#             if layer_id not in params_dict:
#                 params_dict[layer_id] = []
#             params_dict[layer_id].append(param)
#         else:
#             if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
#                 no_decay.append(param)
#             else:
#                 decay.append(param)
#         # add_names.append(name)
#     # print(add_names)
#     sorted_layers = sorted(params_dict.keys(), reverse=True)
#     # 定义分组（示例分为4组）
#     layer_groups = {
#         "deep": sorted_layers[:3],     # 最深3层（如9,10,11）
#         "mid": sorted_layers[3:6],     # 中间3层（如6,7,8）
#         "shallow": sorted_layers[6:9], # 浅层3层（如3,4,5）
#         "input": sorted_layers[9:]     # 输入层（如0,1,2）
#     }
    
#     # 构建参数组
#     param_groups = []
#     for group_name, layers in layer_groups.items():
#         params = []
#         for layer in layers:
#             params.extend(params_dict[layer])
#         param_groups.append({"params": params, "lr": 0.0, 'weight_decay': 0.0, 'group_name': group_name})  # 初始学习率为0
#     return [
#         *param_groups,
#         {'params': no_decay, 'weight_decay': 0., "lr": cfgs.lr},
#         {'params': decay, 'weight_decay': weight_decay, "lr": cfgs.lr}]

def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        opt=cfg.optimizer,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs


def create_optimizer(args, model, filter_bias_and_bn=True):
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
        cfgs=args
    )


def create_optimizer_v2(
        model_or_params,
        opt: str = 'sgd',
        lr: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        filter_bias_and_bn: bool = True,
        cfgs=None,
        **kwargs):
    """ Create an optimizer.

    TODO currently the model is passed in and all parameters are selected for optimization.
    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller

    Args:
        model_or_params (nn.Module): model containing parameters to optimize
        opt: name of optimizer to create
        lr: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through

    Returns:
        Optimizer
    """
    if isinstance(model_or_params, nn.Module):
        # a model was passed in, extract parameters and add weight decays to appropriate layers
        if (weight_decay and filter_bias_and_bn):
            skip = {}
            if hasattr(model_or_params, 'no_weight_decay'):
                skip = model_or_params.no_weight_decay()
            parameters = pre_process(model_or_params, weight_decay, skip, cfgs)
            weight_decay = 0.
        else:
            parameters = model_or_params.parameters()
    else:
        # iterable of parameters or param groups passed in
        parameters = model_or_params

    opt_lower = opt.lower()
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(weight_decay=weight_decay, **kwargs)
    if lr is not None:
        opt_args.setdefault('lr', lr)

    # basic SGD & related
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        # NOTE 'sgd' refers to SGD + nesterov momentum for legacy / backwards compat reasons
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=momentum, nesterov=True, **opt_args)

    # adaptive
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args) 
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'nadam':
        try:
            # NOTE PyTorch >= 1.10 should have native NAdam
            optimizer = optim.Nadam(parameters, **opt_args)
        except AttributeError:
            optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamax':
        optimizer = optim.Adamax(parameters, **opt_args)
    elif opt_lower == 'adabelief':
        optimizer = AdaBelief(parameters, rectify=False, **opt_args)
    elif opt_lower == 'radabelief':
        optimizer = AdaBelief(parameters, rectify=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adagrad':
        opt_args.setdefault('eps', 1e-8)
        optimizer = optim.Adagrad(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'lamb':
        optimizer = Lamb(parameters, **opt_args)
    elif opt_lower == 'lambc':
        optimizer = Lamb(parameters, trust_clip=True, **opt_args)
    elif opt_lower == 'larc':
        optimizer = Lars(parameters, momentum=momentum, trust_clip=True, **opt_args)
    elif opt_lower == 'lars':
        optimizer = Lars(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'nlarc':
        optimizer = Lars(parameters, momentum=momentum, trust_clip=True, nesterov=True, **opt_args)
    elif opt_lower == 'nlars':
        optimizer = Lars(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'madgrad':
        optimizer = MADGRAD(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'madgradw':
        optimizer = MADGRAD(parameters, momentum=momentum, decoupled_decay=True, **opt_args)
    elif opt_lower == 'novograd' or opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=momentum, **opt_args)

    # second order
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)

    # NVIDIA fused optimizers, require APEX to be installed
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)

    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
