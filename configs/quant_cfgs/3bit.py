from easydict import EasyDict as edict

cfg = edict()
BASE_BIT = 3
BASE_ACT_SIGNED = True
BASE_WEIGHT_SIGNED = True
BASE_WEIGHT_CHANNEL_WISE = True
BASE_ACT_CHANNEL_WISE = False
BASE_SCALE_TRAINING = True

# cfg.rounding = None # None for ste
# cfg.rounding = edict(
#         strategy = 'tanh_progressive',
#         initial_hardness = 6.0,
#         final_hardness = 1000.0,
#         num_epochs = 60,
#         schedule = 'exp'       # options: ['linear', 'cosine', 'exp', 'ln']
#     )
cfg.rounding = edict(
        strategy = 'drop_ste',
        initial_drop_rate = 0.99,
        adjust_from_grads = True,
        # max_offset = 0.25,
        drop_epc_ratio=0.5,
        outliers_drop_rate = 0.99,
        final_drop_rate = 0.
    )


# calibration settings
cfg.w_bit = BASE_BIT
cfg.a_bit = BASE_BIT
# cfg.s_bit = BASE_BIT
cfg.w_scale_training = BASE_SCALE_TRAINING
cfg.a_scale_training = BASE_SCALE_TRAINING
cfg.w_channel_wise = BASE_WEIGHT_CHANNEL_WISE
cfg.a_channel_wise = BASE_ACT_CHANNEL_WISE
cfg.w_signed = BASE_WEIGHT_SIGNED
cfg.a_signed = BASE_ACT_SIGNED

cfg.epochs_reset_qparams = 1

cfg.qconv_weight = edict(
        channel_wise=BASE_WEIGHT_CHANNEL_WISE, # options=["channel_wise", "layer_wise"]
        scale_training=False,
        bit_width = 8)

cfg.qconv_act = edict(
        channel_wise=True, # options=["channel_wise", "layer_wise"]
        scale_training=False,
        # signed=BASE_ACT_SIGNED,
        bit_width = 8
    )
cfg.scale_grad = True
# cfg.act_ema_decay = 0.9
cfg.act_ema_decay = 0.9 # TODO: ablation, when act scale-training as False.
cfg.weight_ema_decay = 0.99 # TODO: ablation, when weight scale training as False

cfg.weight_clip_percentile = 99.99
cfg.act_clip_percentile_high = 99.99
cfg.act_clip_percentile_low = 0.01

cfg.post_gelu_quantize = edict(
        type = 'adaptive_log',
        bit_width = BASE_BIT,
        act_clip_percentile_high = 99.99,   # TODO: clip more
        act_clip_percentile_low = 0.01,
        signed = False,
        # channel_wise = BASE_ACT_CHANNEL_WISE
        channel_wise = False,
        enable_adaptive_shift = True,
        scale_training = BASE_SCALE_TRAINING
    )
cfg.post_softmax_quantize = edict(
        type = 'adaptive_log',
        bit_width = BASE_BIT,
        act_clip_percentile_high = 100.,
        act_clip_percentile_low = 0.0,
        signed = False,
        channel_wise = True,    # actually, 'head-wise'
        enable_adaptive_shift = False,
        scale_training = BASE_SCALE_TRAINING
    )
