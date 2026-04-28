from easydict import EasyDict as edict

# MODEL_ZOO={
#     'vit_base_patch16_224_in21k': './checkpoints/ViT-B_16.npz',
#     'vit_large_patch16_224_in21k': './checkpoints/imagenet21k_ViT-L_16.npz'
#     }

MODELS = edict()

MODELS.vitb16 = edict({
    'full_name': 'vit_base_patch16_224_in21k',
    'full_ckpt_path': './checkpoints/ViT-B_16.npz',
    # 'low_bit_ckpt_path': 'ptq_outputs/AdaLog/calib_base_models/{}_{}_{}.pth'   # TODO: make it more elegant. e.g.: vitb16, 2bit, cifar 
    # 'model_2bit_path': './ptq_outputs/AdaLog/6bit/vitb16_vtab/{}/20250317174753/vitb16_w6_a6_s6_calibsize_32_on-{}.pth'   # TODO: make it more elegant
    })

MODELS.vitl = edict({
    'full_name': 'vit_large_patch16_224_in21k',
    'full_ckpt_path': './checkpoints/imagenet21k_ViT-L_16.npz',
    # 'low_bit_ckpt_path': 'ptq_outputs/AdaLog/calib_base_models/{}_{}_{}.pth'   # TODO: make it more elegant. e.g.: vitb16, 2bit, cifar 
    # 'model_2bit_path': './ptq_outputs/AdaLog/6bit/vitb16_vtab/{}/20250317174753/vitb16_w6_a6_s6_calibsize_32_on-{}.pth'   # TODO: make it more elegant
    })