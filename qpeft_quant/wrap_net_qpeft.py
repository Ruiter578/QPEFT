#484行注意改动


import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer, Mlp
from timm.models.swin_transformer import WindowAttention
from types import MethodType
import torch.nn as nn
import math
# from .utils import grad_scale, STEQuant, ProgressiveTanhQuant, DropSTEQuant, get_rounding_func, cupy_quantile
# from .act_quantizer import EMAQuantileMinMaxQuantizer, ActQ, Log2Quantizer, AdaptivePowerLawQuantizer, ShiftLog2Quantizer
from .quantizer import EMAQuantileMinMaxQuantizer, ActQ, Log2Quantizer, AdaptivePowerLawQuantizer, ShiftLog2Quantizer, BaseMinMaxQuantizer, WeightUniformQuantizer
import torch.nn.functional as F
from configs import TUNE_HEAD_PARAMS, TUNE_LORA_PARAMS, TUNE_QUANT_PARAMS
import copy


EPS=1.e-6

class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B

def vit_attn_forward(self, x):
    B, N, C = x.shape
    x = self.qkv(x)
    qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = (qkv[0], qkv[1], qkv[2])
    q, k = self.q_norm(q), self.k_norm(k)
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def swin_attn_forward(self, x, mask=None):
    B_, N, C = x.shape
    x = self.qkv(x)
    qkv = x.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1))
    attn = attn + self._get_rel_pos_bias()
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(-1, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
    

def make_qpeft_vit(transformer_class):
    class ViTQPEFT(transformer_class):

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            return super().forward(*args, **kwdargs)

        def _reset_qparams(self):
            if hasattr(self, 'modules_to_reset'):
                pass
            else:
                modules_to_reset = []
                for name, module in self.named_modules():
                    # if hasattr(module, 'reset_qparams'):    # weight classes
                    if hasattr(module, 'weight_quantizer') and hasattr(module, 'reset_qparams'):    # weight classes
                        if module.weight_quantizer.scale_training:
                            pass
                        else:
                            modules_to_reset.append(module)
                setattr(self, 'modules_to_reset', modules_to_reset)

            for m in self.modules_to_reset:
                m.reset_qparams()

        def _reset_approx_hardness(self, hardness: float):
            if hasattr(self, 'modules_approx_rounding'):
                pass
            else:
                modules_approx_rounding = []
                for name, module in self.named_modules():
                    if hasattr(module, 'reset_hardness'):
                        modules_approx_rounding.append(module)
                setattr(self,'modules_approx_rounding', modules_approx_rounding)

            for m in self.modules_approx_rounding:
                m.reset_hardness(hardness)

        def _reset_rounding_func(self, rounding_cfgs):
            # rounding_cfgs == None for ste
            if hasattr(self,'modules_rounding_strategy'):
                pass
            else:
                modules_rounding_strategy = []
                for name, module in self.named_modules():
                    if hasattr(module, 'reset_rounding_func'):
                        modules_rounding_strategy.append(module)
                setattr(self,'modules_rounding_strategy', modules_rounding_strategy)
            for m in self.modules_rounding_strategy:
                m.reset_rounding_func(rounding_cfgs)

    return ViTQPEFT

def get_act_quantizer(in_channel: int,
                      cfgs,
                      layer_type:str='default',
                      scale_training:bool=False,
                      ):
    rounding_cfgs = cfgs.get('rounding', None)
    if layer_type == 'default':
        act_clip_percentile_high = cfgs.act_clip_percentile_high
        act_clip_percentile_low = cfgs.act_clip_percentile_low
        signed = cfgs.a_signed
        bit_width = cfgs.a_bit
        channel_wise = cfgs.a_channel_wise
    elif layer_type == 'post_gelu':
        bit_width = cfgs.post_gelu_quantize.bit_width
        if bit_width == 32:
            return torch.nn.Identity()
        quantizer_type = getattr(cfgs.post_gelu_quantize, 'type', None)
        act_clip_percentile_high = cfgs.post_gelu_quantize.act_clip_percentile_high
        act_clip_percentile_low = cfgs.post_gelu_quantize.act_clip_percentile_low
        signed = cfgs.post_gelu_quantize.signed
        channel_wise = cfgs.post_gelu_quantize.channel_wise
        if not quantizer_type:
            pass
        elif quantizer_type == 'adaptive_log':
            return AdaptivePowerLawQuantizer(
                bit_width=bit_width,
                channel_wise=channel_wise,
                channel_num=in_channel,
                ema_decay=0. if scale_training else cfgs.act_ema_decay,
                rounding_cfgs=rounding_cfgs,
                log_base_ema=getattr(cfgs.post_gelu_quantize, 'log_base_ema', False),
                enable_adaptive_shift=getattr(cfgs.post_gelu_quantize, 'enable_adaptive_shift', False),
                is_post_gelu=True,
                act_clip_percentile_high=act_clip_percentile_high,
                act_clip_percentile_low=act_clip_percentile_low,
                scale_training=scale_training,
                scale_grad=cfgs.scale_grad
            )
        else:
            raise ValueError

        
    elif layer_type == 'post_softmax':
        quantizer_type = getattr(cfgs.post_softmax_quantize, 'type', None)
        if getattr(cfgs.post_softmax_quantize, 'bit_width') == 32:
            return torch.nn.Identity()
        if not quantizer_type:
            act_clip_percentile_high = cfgs.post_softmax_quantize.act_clip_percentile_high
            act_clip_percentile_low = cfgs.post_softmax_quantize.act_clip_percentile_low
            signed = cfgs.post_softmax_quantize.signed
            bit_width = cfgs.post_softmax_quantize.bit_width
            channel_wise = cfgs.post_softmax_quantize.channel_wise
        elif quantizer_type == 'log2':
            return Log2Quantizer(
                    bit_width=cfgs.post_softmax_quantize.bit_width,
                    channel_wise=cfgs.post_softmax_quantize.channel_wise,
                    channel_num=in_channel,
                    ema_decay=0. if scale_training else cfgs.act_ema_decay,
                    rounding_cfgs=rounding_cfgs,
                    scale_training=scale_training,
                    scale_grad=cfgs.scale_grad
                    )
        elif quantizer_type == 'adaptive_log':
            return AdaptivePowerLawQuantizer(
                    bit_width=cfgs.post_softmax_quantize.bit_width,
                    channel_wise=cfgs.post_softmax_quantize.channel_wise,
                    channel_num=in_channel,
                    ema_decay=0. if scale_training else cfgs.act_ema_decay,
                    rounding_cfgs=rounding_cfgs,
                    log_base_ema=getattr(cfgs.post_softmax_quantize, 'log_base_ema', False),
                    enable_adaptive_shift=getattr(cfgs.post_softmax_quantize, 'enable_adaptive_shift', False),
                    scale_training=scale_training,
                    scale_grad=cfgs.scale_grad
                    )
        elif quantizer_type == 'shift_log2':
            return ShiftLog2Quantizer(
                    bit_width=cfgs.post_softmax_quantize.bit_width,
                    channel_wise=cfgs.post_softmax_quantize.channel_wise,
                    channel_num=in_channel,
                    ema_decay=0. if scale_training else cfgs.act_ema_decay,
                    rounding_cfgs=rounding_cfgs,
                    scale_training=scale_training,
                    scale_grad=cfgs.scale_grad
                    )
            
        else:
            raise ValueError
    elif layer_type == 'conv':
        act_clip_percentile_high = cfgs.act_clip_percentile_high
        act_clip_percentile_low = cfgs.act_clip_percentile_low
        signed = cfgs.a_signed
        bit_width = cfgs.qconv_act.bit_width
        channel_wise = cfgs.qconv_act.channel_wise
        
    else:
        raise ValueError

    if bit_width < 32:
        act_quantizer = EMAQuantileMinMaxQuantizer(
                bit_width=bit_width,
                ema_decay=0. if scale_training else cfgs.act_ema_decay,
                channel_wise=channel_wise,
                channel_num=in_channel,
                signed=signed,
                rounding_cfgs=rounding_cfgs,
                act_clip_percentile_high=act_clip_percentile_high,
                act_clip_percentile_low=act_clip_percentile_low,
                scale_training=scale_training,
                scale_grad=cfgs.scale_grad
            )
    else:
        act_quantizer = torch.nn.Identity()
    return act_quantizer

def make_qpeft_conv2d(conv2d_class, cfgs):
    class QConv2D(conv2d_class):

        def forward(self, x):
            assert self.initialized

            weight = self.weight

            if hasattr(self, 'enable_quantization'):
                assert self.enable_quantization
                assert hasattr(self, 'weight_quantizer')
                
                if self.bit_width < 32:
                    if self.training and self.weight_quantizer.scale_training:
                        if self.weight_quantizer.g == 0.:
                            self.init_quantizer_grad_scale()
                    w_deq = self.weight_quantizer(weight)
                else:
                    w_deq = weight
                    
                x_deq = self.act_quantizer(x)

                return F.conv2d(x_deq, w_deq, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            
        def init_quantizer_grad_scale(self):
            assert self.initialized
            assert getattr(self, 'enable_quantization', False)

            if self.training and self.weight_quantizer.scale_training:
                if self.weight_quantizer.g == 0:    # g does not initialized
                    train_weight_numel = 0
                    if cfgs.scale_grad:
                        for param in self.parameters():
                            if param.requires_grad:
                                train_weight_numel += param.numel()

                    self.weight_quantizer.set_grad_scale(train_weight_numel)    # TODO: frozen + tunable
            else:
                raise ValueError(f"Check scaling grad logic.")
        
        def init_quantize(self):
            if not hasattr(self, 'initialized'):
                self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))
                
            if not hasattr(self, 'enable_quantization'):
                self.register_buffer('enable_quantization', torch.tensor(0, dtype=torch.bool))

            rounding_cfgs = cfgs.get('rounding', None)
            # self.rounding = get_rounding_func(rounding_cfgs)
            qconv_weight_cfgs = cfgs.qconv_weight
            setattr(self, 'bit_width', qconv_weight_cfgs.bit_width)
            # assert not qconv_weight_cfgs.channel_wise
            self.cfgs = cfgs

            self.weight_quantizer = WeightUniformQuantizer(bit_width=qconv_weight_cfgs.bit_width,
                                                            channel_wise=qconv_weight_cfgs.channel_wise,
                                                            channel_num=self.out_channels,
                                                            clip_percentile=cfgs.weight_clip_percentile,
                                                            ema_decay=0. if qconv_weight_cfgs.scale_training else cfgs.weight_ema_decay,
                                                            rounding_cfgs=rounding_cfgs,
                                                            scale_training=cfgs.qconv_weight.scale_training,
                                                            scale_grad=cfgs.scale_grad
                                                            )
            self.act_quantizer = get_act_quantizer(in_channel=self.in_channels,
                                                   cfgs=cfgs,
                                                   layer_type='conv',
                                                   scale_training=cfgs.qconv_act.scale_training
                                                   ) # TODO

            self.initialized.fill_(True)
            self.enable_quantization.fill_(True)

    return QConv2D

@torch.no_grad()
def compute_dsas_scaling_factor(
    W: torch.Tensor, 
    delta_W: torch.Tensor, 
    quantizer_scale: torch.Tensor,
    gamma: float = 0.1,
    c: float = 1.0,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    计算双重尺度自适应缩放 (DSAS) 因子。

    Args:
        W (torch.Tensor): 预训练权重矩阵。
        delta_W (torch.Tensor): LoRA增量矩阵 (B @ A)。
        quantizer_scale (torch.Tensor): 该层量化器的量化步长 (scale)。
                                         注意：可能是标量 (per-tensor) 或向量 (per-channel)。
        gamma (float): 权重稳定性系数，控制 s*delta_W 相对于 W 的最大波动。
        c (float): 量化可见性系数。
        epsilon (float): 防止除以零的小常数。

    Returns:
        torch.Tensor: 计算得到的缩放因子 s_layer。
    """
    # 确保张量在同一设备上
    device = W.device
    delta_W = delta_W.to(device)
    
    # --- 核心计算 ---

    # 1. 计算 W 和 delta_W 的标准差
    #    使用 .detach() 来防止这些计算影响反向传播图
    with torch.no_grad():
        # std_W = torch.std(W).detach()
        # std_delta_W = torch.std(delta_W).detach()
        std_W = torch.norm(W, dim=1).detach()
        std_delta_W = torch.norm(delta_W, dim=1).detach()

    # 2. 计算量化可见性缩放因子 (s_quant)
    #    目标: 让 s * delta_W 的量级与量化步长 alpha 匹配
    #    这里 quantizer_scale 就是 alpha
    s_quant = c * (quantizer_scale / (std_delta_W + epsilon))

    # 3. 计算权重稳定性缩放因子 (s_stable)
    #    目标: 限制 s * delta_W 的量级，使其不超过 W 量级的一个比例 (gamma)
    s_stable = gamma * (std_W / (std_delta_W + epsilon))
    
    # 4. 取两者中的较小值，以同时满足两个约束
    #    使用 torch.min 来处理 per-channel 的情况
    s_layer = torch.min(s_quant, s_stable)

    return s_layer

def make_qpeft_linear(linear_class, cfgs):
    class QPEFTLinear(linear_class):

        def forward(self, x):
            assert self.initialized

            orig_weight = self.weight

            if hasattr(self, 'enable_lora'):
                assert self.enable_lora
                assert hasattr(self, 'lora_A') and hasattr(self, 'lora_B')
                weight = self.get_merged_weight(orig_weight, self.lora_A, self.lora_B)
            else:
                weight = orig_weight

            if hasattr(self, 'enable_quantization'):
                assert self.enable_quantization

                if self.bit_width < 32:
                    if self.training and self.weight_quantizer.scale_training:
                        if self.weight_quantizer.g == 0.:
                            self.init_quantizer_grad_scale()
                    w_deq = self.weight_quantizer(weight)
                else:
                    w_deq = weight
                    
                x_deq = self.act_quantizer(x)

                return F.linear(x_deq, w_deq, self.bias)
            else:
                return F.linear(x, weight, self.bias)
            
        def init_quantizer_grad_scale(self):
            assert self.initialized
            assert getattr(self, 'enable_quantization', False)

            if self.training and self.weight_quantizer.scale_training:
                if self.weight_quantizer.g == 0:    # g does not initialized
                    train_weight_numel = 0
                    if cfgs.scale_grad:
                        for param in self.parameters():
                            if param.requires_grad:
                                train_weight_numel += param.numel()

                    self.weight_quantizer.set_grad_scale(train_weight_numel)    # TODO: frozen + tunable
            else:
                raise ValueError(f"Check scaling grad logic.")
        
        def calculate_scale(self, weight):
            assert hasattr(self, 'layer_type')
            assert weight.shape == self.weight.shape
            weight_clip_percentile = cfgs.weight_clip_percentile / 100.
            q_max = self.weight_quantizer.q_max
            q_min = self.weight_quantizer.q_min
            EPS = self.weight_quantizer.EPS

            if cfgs.qconv_weight.channel_wise:
                if cfgs.weight_clip_percentile < 100.:
                    max_val_pos = torch.quantile(weight.abs(), dim=1, q=weight_clip_percentile)
                else:
                    max_val_pos = torch.max(weight.abs(), dim=1)[0]  # co
                    
            else:
                if cfgs.weight_clip_percentile < 100.:
                    max_val_pos = torch.quantile(weight.abs(), q=weight_clip_percentile)
                else:
                    max_val_pos = torch.max(weight.abs())[0]
            alpha = max_val_pos / ((q_max - q_min) / 2)
            alpha = torch.max(alpha, EPS)
            return alpha

        def init_quantize(self):
            if not hasattr(self, 'initialized'):
                self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))

            if not hasattr(self, 'enable_quantization'):
                self.register_buffer('enable_quantization', torch.tensor(0, dtype=torch.bool))

            assert hasattr(self, 'layer_type')
            
            setattr(self, 'bit_width', cfgs.w_bit)
            self.cfgs = cfgs

            self.weight_quantizer = WeightUniformQuantizer(bit_width=cfgs.w_bit,
                                                            channel_wise=cfgs.w_channel_wise,
                                                            channel_num=self.out_features,
                                                            clip_percentile=cfgs.weight_clip_percentile,
                                                            ema_decay=0. if cfgs.w_scale_training else cfgs.weight_ema_decay,
                                                            rounding_cfgs=cfgs.rounding,
                                                            scale_training=cfgs.w_scale_training,
                                                            scale_grad=cfgs.scale_grad
                                                        )
            

            if self.layer_type == 'default':
                a_scale_training = cfgs.a_scale_training
            elif self.layer_type == 'post_gelu':
                a_scale_training = cfgs.post_gelu_quantize.scale_training
            else:
                raise ValueError
            self.act_quantizer = get_act_quantizer(in_channel=self.in_features,
                                                   cfgs=cfgs,
                                                   layer_type=self.layer_type,
                                                   scale_training=a_scale_training
                                                   ) # TODO

            self.initialized.fill_(True)
            self.enable_quantization.fill_(True)

        @torch.no_grad()
        def init_lora(self):
            if not hasattr(self, 'initialized'):
                self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))
            if not hasattr(self, 'enable_lora'):
                self.register_buffer('enable_lora', torch.tensor(0, dtype=torch.bool))
            assert cfgs.lora_cfgs is not None
            # self.lora_A =  nn.Parameter(nn.init.kaiming_uniform(torch.zeros([self.in_features, cfgs.lora_r]), a = math.sqrt(5)))
            self.lora_A = nn.Parameter(torch.zeros([self.in_features, cfgs.lora_r]))
            self.lora_B = nn.Parameter(torch.zeros([cfgs.lora_r, self.out_features]))
            nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))

            lora_scale_val = getattr(cfgs, 'lora_scale', 1.0)
            if getattr(cfgs, 'lora_scale_train', False):
                self.lora_scale = nn.Parameter(torch.full([1], lora_scale_val))
            else:
                self.register_buffer('lora_scale', torch.full([1], lora_scale_val))
            self.lora_type =  getattr(cfgs, 'type', 'lora')
            assert self.lora_type in ['lora', 'dora']
            if self.lora_type == 'dora':
                weight = self.weight
                lora_weight = self.get_composed_lora_weight(self.lora_A, self.lora_B)
                weight_norm = self.get_weight_norm(weight + lora_weight * self.lora_scale)  # init from full-precision weights
                self.magnitude = nn.Parameter(weight_norm)

            # 注意改动：if cfgs.quant_cfgs is not None:
            #     self.register_buffer('scale_ema', torch.ones(self.out_features))
            self.initialized.fill_(True)
            self.enable_lora.fill_(True)

        def get_composed_lora_weight(self, A, B):
            return torch.einsum("ir, ro -> oi", A, B)

        # def get_merged_weight(self, weights, lora_A, lora_B):
        #     if self.lora_type == 'lora':
        #         delta_w = self.get_composed_lora_weight(lora_A, lora_B)
        #         s = compute_dsas_scaling_factor(weights, delta_w, self.weight_quantizer.alpha.mean(), c=0.1)
        #         # import pdb; pdb.set_trace()
        #         # if torch.all(self.scale_ema == -1):
        #         #     self.scale_ema.copy_(s)
        #         # else:
        #         #     self.scale_ema.copy_(self.scale_ema * 0.95 + s * 0.05)
        #         # self.scale_ema.copy_(self.scale_ema * 0.9 + s * 0.1)
        #         # assert delta_w.shape == weights.shape
        #         # return weights + delta_w * self.lora_scale * self.scale_ema[:, None]
            
        #     elif self.lora_type == 'dora':
        #         magnitude = self.magnitude  # maintain full-precision
        #         delta_w = self.get_composed_lora_weight(lora_A, lora_B)

        #         merged_unscaled_weight = weights + delta_w * self.lora_scale
        #         with torch.no_grad():

        #             weight_norm = self.get_weight_norm(merged_unscaled_weight)
        #         # mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        #         mag_norm_scale = (magnitude / weight_norm).view(-1, 1)  # [c_out, 1] for matching weight dimension

        #         return mag_norm_scale * merged_unscaled_weight

        #     else:
        #         raise ValueError(f"Unknown lora type {self.lora_type}")
        def get_merged_weight(self, weights, lora_A, lora_B):
            if self.lora_type == 'lora':
                delta_w = self.get_composed_lora_weight(lora_A, lora_B)

                assert delta_w.shape == weights.shape
                return weights + delta_w * self.lora_scale
            
            elif self.lora_type == 'dora':
                magnitude = self.magnitude  # maintain full-precision
                delta_w = self.get_composed_lora_weight(lora_A, lora_B)

                merged_unscaled_weight = weights + delta_w * self.lora_scale
                with torch.no_grad():

                    weight_norm = self.get_weight_norm(merged_unscaled_weight)
                # mag_norm_scale = (magnitude / weight_norm).view(1, -1)
                mag_norm_scale = (magnitude / weight_norm).view(-1, 1)  # [c_out, 1] for matching weight dimension

                return mag_norm_scale * merged_unscaled_weight

            else:
                raise ValueError(f"Unknown lora type {self.lora_type}")

        def get_weight_norm(self, weight):
            """
            Calculate the L2 norm of the merged weight matrix column-wise. This method is adapted from the implementation in Hugging Face's PEFT library.

            Args:
                weight (torch.Tensor): [c_out, c_in]. The original weight matrix.

            Returns:
                torch.Tensor: [c_out]. The L2 norm of the merged weight matrix calculated column-wise, with the same data type as the input weight.
            """
            # Adapt from https://github.com/huggingface/peft/blob/59ef3b93c8feda05fa92d8de7d588c30907266b5/src/peft/tuners/lora/dora.py#L30
            # Merge the original weight with the LoRA weight by adding the LoRA weight multiplied by the scaling factor
            # weight = weight + lora_scale * lora_weight
            # Calculate the L2 norm of the merged weight matrix column-wise and convert the result data type to match the input weight
            weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
            return weight_norm  # [c_out] actually, different with the paper. Ref. https://github.com/NVlabs/DoRA/issues/11
            
        def extra_repr(self):
            s_prefix = super().extra_repr()
            if hasattr(self, 'enable_lora'):
                assert self.enable_lora
                assert hasattr(self, 'lora_A')
                assert hasattr(self, 'lora_B')
                s_prefix += f", lora_A.shape={list(self.lora_A.shape)}, lora_B.shape={list(self.lora_B.shape)}, lora_type={self.lora_type}" 
                if self.lora_type == 'dora':
                    s_prefix += f", magnitude.shape={list(self.magnitude.shape)}"
            return s_prefix

        @torch.no_grad()
        def reset_qparams(self):
            assert getattr(self, 'enable_lora', False) and getattr(self, 'enable_quantization', False), f"enable_lora={self.enable_lora}, enable_quantization={self.enable_quantization}"
            if isinstance(self.weight_quantizer.alpha, nn.Parameter):
                return  # do not manually update alpha when it is learnable

            weight = self.get_merged_weight(self.weight + self.get_composed_lora_weight(self.lora_A, self.lora_B) * self.lora_scale)

            alpha = self.calculate_scale(weight)
            self.weight_quantizer.alpha.copy_(alpha.clone().detach())
            
    return QPEFTLinear

def make_quant_attn(attn_class, cfgs):
    class QAttention(attn_class):

        def forward(self, x):
            B, N, C = x.shape
            x = self.qkv(x)
            qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = (qkv[0], qkv[1], qkv[2])
            q, k = self.q_norm(q), self.k_norm(k)
            attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = self.matmul2(attn, v)
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        def matmul1(self, q, k, k_transposed=True):
            # q: [B, num_heads, N, head_dim]
            # K: [B, num_heads, head_dim, N] or [B, num_heads, N, head_dim]
            q = self.q_quantizer(q)
            if k_transposed:
                k = self.k_quantizer(k.transpose(-2, -1)).transpose(-2, -1)
            else:
                k = self.k_quantizer(k).transpose(-2, -1)
            return q @ k

        def matmul2(self, attn, v):
            return self.attn_quantizer(attn) @ self.v_quantizer(v)
        
    
        def init_quantize(self):
            if not hasattr(self, 'initialized'):
                self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))
                
            if not hasattr(self, 'enable_quantization'):
                self.register_buffer('enable_quantization', torch.tensor(0, dtype=torch.bool))

            act_quant_default = get_act_quantizer(
                                    in_channel=self.num_heads,  # TODO, check num_heads or head_dim
                                    cfgs=cfgs,
                                    layer_type='default',
                                    scale_training=cfgs.a_scale_training
                    )
            self.q_quantizer = copy.deepcopy(act_quant_default)
            self.k_quantizer = copy.deepcopy(act_quant_default)
            self.v_quantizer = copy.deepcopy(act_quant_default)

            post_sm_cfg = cfgs.post_softmax_quantize
            self.attn_quantizer = get_act_quantizer(
                        in_channel=self.num_heads,
                        cfgs=cfgs,
                        layer_type='post_softmax',
                        scale_training=post_sm_cfg.scale_training
                )

    return QAttention

def apply_qpeft(model: VisionTransformer, cfgs):
    QVisionTransformer = make_qpeft_vit(model.__class__)

    model.__class__ = QVisionTransformer

    for name, module in model.named_modules():
        if isinstance(module, Attention):
            if cfgs.quant_cfgs is not None:
                module.__class__ = make_quant_attn(module.__class__, cfgs)
                module.init_quantize()
            
    module_dict={}
    for name, module in model.named_modules():
        module_dict[name] = module
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(module, nn.Conv2d):
            if cfgs.quant_cfgs is not None: # TODO: enable lora for patch embed?
                module.__class__ = make_qpeft_conv2d(module.__class__, cfgs)
                module.init_quantize()
        elif isinstance(module, nn.Linear):
            layer_type = 'default'
            wrap_class = False
            init_quant = False
            init_lora = False
            if 'head' in name:
                continue
            # elif 'mlp.' in name:
            #     pass
            elif 'fc2' in name:
                layer_type = 'post_gelu'
                
            if cfgs.quant_cfgs is not None: # do quantization
                # module.__class__ = make_qpeft_linear(module.__class__, cfgs)
                wrap_class = True
                init_quant = True
            # else:   # lora
            if cfgs.lora_cfgs is None:
                init_lora = False
            else:
                lora_apply_pos = getattr(cfgs, 'lora_apply_position', None)
                assert isinstance(lora_apply_pos, (str, list, type(None)))
                if isinstance(lora_apply_pos, str):
                    lora_apply_pos = [lora_apply_pos]

                for pos in lora_apply_pos:
                    if pos is None:  # default for all layers
                        wrap_class = True
                        init_lora = True
                    elif pos in name:
                        wrap_class = True
                        init_lora = True

            if wrap_class:
                module.__class__ = make_qpeft_linear(module.__class__, cfgs)
                setattr(module, 'layer_type', layer_type)
            else:
                print(f"Does not apply LoRA/Quant to {name}")
            if init_quant:
                module.init_quantize()

            if init_lora:
                module.init_lora()



def prepare_for_peft(model, logger, cfgs):    
    # wrap_quant_params(model, qmethod='AdaLog')  # TODO: support more quantization method
    # wrap_lora_in_attn(model, args)

    quant_tune_msg, lora_tune_msg, head_tune_msg = '', '', ''
    freeze_msg = ''
    total_quant_num = 0
    quant_tune_num, lora_tune_num, head_tune_num = 0, 0, 0
    freeze_num, total_num, total_tuning_num = 0, 0, 0
    for name, params in model.named_parameters():
        if name.endswith(TUNE_QUANT_PARAMS):
            if 'patch_embed.' in name:
                if cfgs.qconv_weight.scale_training:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            elif 'log_base_alpha' in name or 'adaptive_shift' in name:
                params.requires_grad = True
            else:
                if cfgs.w_scale_training:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            quant_tune_msg += f"{name}: {params.shape}, Trainable = {params.requires_grad}\n"
            quant_tune_num += params.numel()
            total_quant_num += params.numel()
        elif name.endswith(TUNE_LORA_PARAMS):
            params.requires_grad = True
            lora_tune_msg += f"{name}: {params.shape}\n"
            lora_tune_num += params.numel()
        elif name.endswith(TUNE_HEAD_PARAMS):
            params.requires_grad = True
            head_tune_msg += f"{name}: {params.shape}\n"
            head_tune_num += params.numel()
        else:
            params.requires_grad = False
            
        if params.requires_grad == False:
            freeze_msg += f"{name}: {params.shape}\n"
            freeze_num += params.numel()
        else:
            total_tuning_num += params.numel()
        total_num += params.numel()

    logger.info(f"Quant Parameters are:\n" + quant_tune_msg)
    logger.info(f"LoRA Tuning Parameters are:\n" + lora_tune_msg)
    logger.info(f"Head Tuning Parameters are:\n" + head_tune_msg)
    logger.info(f"Frozen Parameters are:\n" + freeze_msg)

    sum_msg = "==="*20 +"\nSummary: " + f"Total Quant: {total_quant_num}, Quant Tuning: {quant_tune_num}, LoRA: {lora_tune_num},\n" + \
    f"Quant+LoRA Tuning: {quant_tune_num+lora_tune_num}, Head: {head_tune_num}, Frozen: {freeze_num}, Total: {total_num}, Total Tuning: {total_tuning_num}"
    logger.info(sum_msg)
