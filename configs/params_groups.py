TUNE_QUANT_PARAMS = ('_quantizer.scale', '.zero_point', '.alpha', '.log_base_alpha', '.adaptive_shift')
# TUNE_QUANT_PARAMS = ('_quantizer.scale')
TUNE_LORA_PARAMS = ('.lora_A', '.lora_B', '.magnitude', '.lora_scale')
TUNE_HEAD_PARAMS = ('head.weight', 'head.bias')

# TODO: support configuration.