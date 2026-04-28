import os
import torch
from collections import OrderedDict
import torch.nn as nn

def load_state_dict(checkpoint_path, use_ema=False, moco=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        if state_dict_key:
            state_dict = checkpoint[state_dict_key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        if moco:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # strip `base_encoder.` prefix
                name = k[13:] if k.startswith('base_encoder') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            pass
        # logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        # logger.info("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def timm_load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    if "mocov3" in checkpoint_path:
        moco = True
    else:
        moco = False
    state_dict = load_state_dict(checkpoint_path, use_ema, moco)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    return missing_keys, unexpected_keys

def load_quant_state_dict(model, ckpt_path):
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.calibrated = True
            module.mode = 'quant_forward'
        if isinstance(module, nn.Linear) and 'reduction' in name:
            module.bias = nn.Parameter(torch.zeros(module.out_features))
        quantizer_attrs = ['a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer']
        for attr in quantizer_attrs:
            if hasattr(module, attr):
                getattr(module, attr).inited = True
    ckpt = torch.load(ckpt_path)
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)

    return missing_keys, unexpected_keys