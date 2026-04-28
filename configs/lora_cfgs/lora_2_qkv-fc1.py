from easydict import EasyDict as edict

cfg = edict()

cfg.lora_r = 2
cfg.lora_apply_position = ['qkv', 'fc1']