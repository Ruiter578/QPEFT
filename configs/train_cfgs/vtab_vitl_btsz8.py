from easydict import EasyDict as edict

cfg = edict()

cfg.data_path = './data/vtab-1k'
cfg.batch_size = 8
cfg.optimizer = 'AdamW'
cfg.lr = 0.0005
cfg.min_lr = 1e-7
cfg.epochs = 100
cfg.warmup_epochs = 20
cfg.num_workers = 10
cfg.weight_decay = 1e-4
cfg.momentum = 0.9
cfg.drop_path = 0.1
cfg.val_interval = 10

# cfg.qtuning_start_epc = 20
# cfg.qtuning_lr_ratio = 0.05
# cfg.progress_optim_by_depth = True
# cfg.unfreeze_schedule = {
#     "deep": 5,    # 初始已解冻
#     "mid": 10,     # 第10个epoch解冻
#     "shallow": 15, # 第20个epoch解冻
#     "input": 20    # 第30个epoch解冻
# }
# cfg.unfreeze_lr_ratio = {
#     "deep": 0.8,    # 初始已解冻
#     "mid": 0.3,     # 第10个epoch解冻
#     "shallow": 0.1, # 第20个epoch解冻
#     "input": 0.05    # 第30个epoch解冻
# }