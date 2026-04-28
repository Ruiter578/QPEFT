# TODO: It is more like log them uniform quantizer. 
import torch
from .utils import (calculate_qminmax, grad_scale, round_pass, cupy_quantile,
                    get_rounding_func)
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cupy

EPS=1.e-6   # TODO: maybe too small for post-softmax?

class BaseQuantizer(nn.Module):
    """Base class for quantization modules."""
    
    def __init__(self, bit_width: int, channel_wise: bool, channel_num: int, rounding_cfgs: dict = None, scale_training: bool = False, scale_grad=True):
        """
        Args:
            bit_width: Number of bits for quantization
            channel_wise: Whether to apply channel-wise quantization
            channel_num: Number of channels for channel-wise quantization
            rounding_strategy: Rounding method ('ste' or 'tanh')
        """
        super().__init__()
        assert 0 < bit_width <= 32
        self.bit_width = bit_width
        self.channel_wise = channel_wise
        self.channel_num = channel_num
        self.scale_training = scale_training
        self.scale_grad = scale_grad
        
        self.reset_rounding_func(rounding_cfgs)

    def reset_rounding_func(self, rounding_cfgs):
        """
        Set the rounding strategy for quantization.

        Args:
            rounding_strategy: Rounding method ('ste' or 'tanh')
        """
        self.rounding = get_rounding_func(rounding_cfgs)
        
    def _flatten_to_2d(self, x: torch.Tensor) -> tuple:
        """Flatten input to 2D tensor (batch_size*..., channels)."""
        orig_shape = x.shape
        if len(orig_shape) == 4:
            b, c, h, w = orig_shape
            x = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        elif len(orig_shape) == 3:
            b, n, c = orig_shape
            x = x.reshape(b * n, c)
        elif len(orig_shape) == 2:
            pass
        else:
            raise ValueError(f"Unsupported input shape: {orig_shape}")
        return x, orig_shape

    def _unflatten_to_orig(self, x: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        """Restore original shape from 2D tensor."""
        if len(orig_shape) == 4:
            b, c, h, w = orig_shape
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        elif len(orig_shape) == 3:
            b, n, c = orig_shape
            x = x.reshape(b, n, c).contiguous()
        elif len(orig_shape) == 2:
            x = x.reshape(orig_shape)
        return x

    def _update_stats(self, x: torch.Tensor):
        """Update running statistics (to be implemented by subclasses)."""
        raise NotImplementedError

    def _quantize(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """Quantize input (to be implemented by subclasses)."""
        raise NotImplementedError

    def _dequantize(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize input (to be implemented by subclasses)."""
        raise NotImplementedError

    def set_grad_scale(self, sample_numel):
        if sample_numel == 0:
            g = 1.0
        else:
            g = 1.0 / math.sqrt(sample_numel * self.q_max.item())
        self.g.copy_(torch.tensor(g))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with shape handling."""
        orig_dtype = x.dtype
        x_flat, orig_shape = self._flatten_to_2d(x)

        if self.scale_training:
            if torch.all(self.alpha == 0.): # scale not initialized
                assert not self.training
                cur_min, cur_max = self._calculate_running_minmax(x_flat.detach())
                init_alpha, _ = self._calculate_qparams(cur_min, cur_max)
                self.alpha.copy_(init_alpha)
                # self.zero_point.copy_(init_zp)
                    
        if self.training:
            if self.scale_training: # TODO: does not impl. for scale_grad=False
                scale = self.alpha
                if self.scale_grad:
                    if self.g == 0.:
                        sample_numel = x[0].numel() # numel for each sample
                        self.set_grad_scale(sample_numel)
                    scale = grad_scale(scale, self.g)
                zero_point = self.zero_point
            else:
                self._update_stats(x_flat)
                scale, zero_point = self.alpha, self.zero_point
        else:
            scale, zero_point = self.alpha, self.zero_point
        x_q = self._quantize(x_flat, scale, zero_point)
        x_deq = self._dequantize(x_q, scale, zero_point)
        x_deq = self._unflatten_to_orig(x_deq, orig_shape)
        return x_deq.to(orig_dtype)

    def extra_repr(self) -> str:
        return f"bit_width={self.bit_width}, channel_wise={self.channel_wise}, scale_training={self.scale_training}, scale_grad={self.scale_grad}"


    def _calculate_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor) -> tuple:
        """Calculate scale and zero-point parameters."""
        # Handle non-channel-wise case
        if not self.channel_wise and (max_val - min_val).abs().sum() < self.EPS:
            raise ValueError("max_val == min_val")

        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        if self.signed:
            effective_range = torch.max(-min_val_neg, max_val_pos)
            scale = effective_range / ((self.q_max.float() - self.q_min.float()) / 2)   # Following torch. ref. https://github.com/pytorch/pytorch/blob/e15848669f84d3767bfca724a29a6a6dde3308b9/torch/ao/quantization/observer.py#L379
            zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=min_val_neg.device)
        else:
            scale = (max_val_pos - min_val_neg) / (self.q_max.float() - self.q_min.float())
            zero_point = self.q_min.float() - torch.round(min_val_neg / scale).to(torch.int64)
            zero_point = zero_point.clamp(self.q_min.float(), self.q_max.float())
        
        scale = torch.max(scale, self.EPS.float())
        
        # zero_point = self.q_min.float() - (min_val_neg / scale)
        # zero_point = torch.round(zero_point).clamp(self.q_min.float(), self.q_max.float())
        
        return scale, zero_point

class BaseMinMaxQuantizer(BaseQuantizer):
    """Min/max based quantizer with EMA statistics."""
    
    def __init__(self, bit_width: int, signed: bool, channel_wise: bool, channel_num: int,
                 ema_decay: float = 0.95, rounding_cfgs: dict = None, scale_training: bool = False, scale_grad = True):
        """
        Args:
            signed: Whether to use signed quantization
            ema_decay: Decay rate for EMA statistics
        """
        super().__init__(bit_width, channel_wise, channel_num, rounding_cfgs, scale_training, scale_grad)
        self.signed = signed
        self.ema_decay = ema_decay

        # Set quantization range
        q_min_val, q_max_val = calculate_qminmax(bit_width, signed)
        self.register_buffer('q_min', torch.tensor(q_min_val, dtype=torch.int64))
        self.register_buffer('q_max', torch.tensor(q_max_val, dtype=torch.int64))

        # Initialize buffers
        if channel_wise:
            assert channel_num > 0
            dummy = torch.zeros(channel_num)
        else:
            dummy = torch.tensor(0.)

        self.register_buffer('zero_point', torch.full_like(dummy, 
                           0 if signed else (q_max_val + 1) // 2, dtype=torch.int64))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))
        self.register_buffer('EPS', torch.tensor(EPS))
        
        if not scale_training:
            self.register_buffer('running_min', dummy.clone())
            self.register_buffer('running_max', dummy.clone())
            self.register_buffer('alpha', torch.ones_like(dummy))
        else:
            self.alpha = nn.Parameter(torch.zeros_like(dummy))   # initialize after 
            self.register_buffer('g', torch.tensor(0.))

    def _calculate_running_minmax(self, x: torch.Tensor) -> tuple: # TODO: change to _calculate_running_minmax
        """Calculate min/max values (to be implemented by subclasses)."""
        raise NotImplementedError

    def _update_stats(self, x: torch.Tensor):
        """Update running min/max statistics using EMA."""
        assert not self.scale_training
        cur_min, cur_max = self._calculate_running_minmax(x.detach())
        
        if not self.initialized:
            self.running_min.copy_(cur_min)
            self.running_max.copy_(cur_max)
            self.initialized.fill_(True)
        else:
            self.running_min = self.running_min * self.ema_decay + cur_min * (1 - self.ema_decay)
            self.running_max = self.running_max * self.ema_decay + cur_max * (1 - self.ema_decay)

        alpha, zero_point = self._calculate_qparams(self.running_min, self.running_max)
        self.alpha.copy_(alpha)
        self.zero_point.copy_(zero_point)

    def _quantize(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """Quantize input using scale and zero-point."""
        return self.rounding((x / scale + zero_point).clamp(self.q_min, self.q_max))

    def _dequantize(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize input using scale and zero-point."""
        return (x - zero_point) * scale

    def extra_repr(self) -> str:
        if not self.scale_training:
            return super().extra_repr() + f", signed={self.signed}, ema_decay={self.ema_decay:.3f}"
        return super().extra_repr() + f", signed={self.signed}"


class EMAMinMaxQuantizer(BaseMinMaxQuantizer):
    """Min/max quantizer using exact min/max statistics."""
    
    def _calculate_running_minmax(self, x: torch.Tensor) -> tuple:
        """Compute min/max values."""
        if self.channel_wise:
            return torch.aminmax(x, dim=0)
        return torch.aminmax(x)


class EMAQuantileMinMaxQuantizer(BaseMinMaxQuantizer):
    """Min/max quantizer using percentile-based statistics."""
    
    def __init__(self, bit_width: int, signed: bool, channel_wise: bool, channel_num: int,
                 ema_decay: float = 0.95, rounding_cfgs: dict = None,
                 act_clip_percentile_high: float = 99.99, act_clip_percentile_low: float = 0.01, scale_training: bool = False, scale_grad = True):
        """
        Args:
            act_clip_percentile_high: Upper percentile for clipping
            act_clip_percentile_low: Lower percentile for clipping
        """
        super().__init__(bit_width, signed, channel_wise, channel_num, ema_decay, rounding_cfgs, scale_training, scale_grad)
        assert act_clip_percentile_high <= 100. and act_clip_percentile_low >= 0., "range in [0, 100]"

        self.register_buffer('act_clip_percentile_high', torch.tensor(act_clip_percentile_high / 100.0))
        self.register_buffer('act_clip_percentile_low', torch.tensor(act_clip_percentile_low / 100.0))
    @torch.no_grad()
    def _calculate_running_minmax(self, x: torch.Tensor) -> tuple:
        """Compute percentile-based min/max values."""
        q = torch.tensor([self.act_clip_percentile_low, self.act_clip_percentile_high], device=x.device)
        
        if self.channel_wise:
            if self.act_clip_percentile_high == 1.0 and self.act_clip_percentile_low == 0.0:
                return torch.aminmax(x, dim=0)
            return torch.quantile(x, q=q, dim=0, keepdim=False)
        
        if self.act_clip_percentile_high == 1.0 and self.act_clip_percentile_low == 0.0:
            return torch.aminmax(x)
        return cupy_quantile(x, q=q)



class BaseLogQuantizer(BaseQuantizer):
    """Base class for log-domain quantizers."""
    
    def __init__(self, bit_width: int, channel_wise: bool, channel_num: int,
                 ema_decay: float, act_clip_percentile_high: float,
                 act_clip_percentile_low: float, rounding_cfgs: dict = None, scale_training: bool = False, scale_grad = True):
        """Initialize with quantization parameters."""
        super().__init__(bit_width, channel_wise, channel_num, rounding_cfgs, scale_training, scale_grad)
        self.signed = False # log domain always > 0
        self.ema_decay = ema_decay
        # self.act_clip_percentile_high = act_clip_percentile_high
        # self.act_clip_percentile_low = act_clip_percentile_low

        # Initialize buffers
        if channel_wise:
            dummy = torch.zeros(channel_num)
        else:
            dummy = torch.tensor(0.)
        q_min_val, q_max_val = calculate_qminmax(bit_width, signed=False)

        # Set quantization range (unsigned)
        self.register_buffer('q_min', torch.tensor(q_min_val, dtype=torch.int64))
        self.register_buffer('q_max', torch.tensor(q_max_val, dtype=torch.int64))
        self.register_buffer('EPS', torch.tensor(EPS))
        self.register_buffer('act_clip_percentile_high', torch.tensor(act_clip_percentile_high / 100.0))    # for init or ema
        self.register_buffer('act_clip_percentile_low', torch.tensor(act_clip_percentile_low / 100.0))     
        if not scale_training:
            self.register_buffer('running_min', dummy.clone())
            self.register_buffer('running_max', dummy.clone())
            self.register_buffer('alpha', torch.ones_like(dummy))

        else:
            self.alpha = nn.Parameter(torch.zeros_like(dummy))
            self.register_buffer('g', torch.tensor(0.))
        
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))

    def _transform_to_log(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input to log domain (to be implemented by subclasses)."""
        raise NotImplementedError

    def _transform_from_log(self, y: torch.Tensor) -> torch.Tensor:
        """Transform from log domain to linear (to be implemented by subclasses)."""
        raise NotImplementedError

    def _calculate_running_minmax(self, y: torch.Tensor) -> tuple:
        """Calculate min/max in log domain."""
        # if self.scale_training:
        #     raise ValueError("scale training do not need to calculate running minmax")
        q = torch.tensor([self.act_clip_percentile_low, self.act_clip_percentile_high], device=y.device)
        
        if self.channel_wise:
            if self.act_clip_percentile_high == 1.0 and self.act_clip_percentile_low == 0.0:
                return torch.aminmax(y, dim=0)
            return torch.quantile(y, q=q, dim=0, keepdim=False)
        
        if self.act_clip_percentile_high == 1.0 and self.act_clip_percentile_low == 0.0:
            return torch.aminmax(y)
        return cupy_quantile(y, q=q)

    def _update_stats(self, y: torch.Tensor):
        """Update running min/max statistics in log domain."""
        if self.scale_training:
            raise ValueError("scale training do not need to update running statistics")
        cur_min, cur_max = self._calculate_running_minmax(y.detach())
        
        if not self.initialized:
            self.running_min.copy_(cur_min)
            self.running_max.copy_(cur_max)
            self.initialized.fill_(True)
        else:
            self.running_min = self.running_min * self.ema_decay + cur_min * (1 - self.ema_decay)
            self.running_max = self.running_max * self.ema_decay + cur_max * (1 - self.ema_decay)
        
        # Update scale parameter
        val_range = self.running_max - self.running_min
        scale = val_range / (self.q_max - self.q_min)
        self.alpha.copy_(torch.clamp(scale, min=self.EPS))

    def _quantize(self, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Quantize in log domain."""
        if self.scale_training:
            y_scaled = y / scale + self.q_min
        else:
            y_scaled = (y - self.running_min) / scale + self.q_min
        y_quant = self.rounding(y_scaled)
        return torch.clamp(y_quant, self.q_min, self.q_max)

    def _dequantize(self, y_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize in log domain."""
        if self.scale_training:
            return (y_quant - self.q_min) * scale
        else:
            return (y_quant - self.q_min) * scale + self.running_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with log transformation."""
        orig_dtype = x.dtype
        x_flat, orig_shape = self._flatten_to_2d(x)
        y = self._transform_to_log(x_flat)

        if self.scale_training:
            if torch.all(self.alpha == 0.): # scale not initialized
                assert not self.training
                cur_min, cur_max = self._calculate_running_minmax(y.detach())
                init_scale, _ = self._calculate_qparams(cur_min, cur_max)
                self.alpha.copy_(init_scale)
                # self.zero_point.copy_(init_zp)
                
        if self.training:
            if self.scale_training: # TODO: does not impl. for scale_grad=False
                scale = self.alpha
                if self.scale_grad:
                    if self.g == 0.:
                        sample_numel = x[0].numel()
                        self.set_grad_scale(sample_numel)
                    scale = grad_scale(scale, self.g)
            else:
                self._update_stats(y.detach())
                scale = self.alpha
        else:
            scale = self.alpha
        y_q = self._quantize(y, scale)
        y_deq = self._dequantize(y_q, scale)
        x_deq = self._transform_from_log(y_deq)
        x_deq = self._unflatten_to_orig(x_deq, orig_shape)
        return x_deq.to(orig_dtype)

    # def extra_repr(self) -> str:
    #     return (super().extra_repr() + 
    #             f", ema_decay={self.ema_decay:.3f}, "
    #             f"clip_low={self.act_clip_percentile_low*100}%, "
    #             f"clip_high={self.act_clip_percentile_high*100}%")



class Log2Quantizer(BaseLogQuantizer):
    """Base log2 quantizer."""
    
    def __init__(self, bit_width: int, channel_wise: bool, channel_num: int,
                 ema_decay: float = 0.95, rounding_cfgs: dict = None,
                 act_clip_percentile_high: float = 100.0, act_clip_percentile_low: float = 0.0, scale_training: bool = False, scale_grad = True):
        super().__init__(bit_width, channel_wise, channel_num, ema_decay,
                         act_clip_percentile_high, act_clip_percentile_low, rounding_cfgs, scale_training, scale_grad)

    def _transform_to_log(self, x: torch.Tensor) -> torch.Tensor:
        """Transform to base-2 log domain."""
        return -torch.log2(x.clamp(min=self.EPS))

    def _transform_from_log(self, y: torch.Tensor) -> torch.Tensor:
        """Transform from log domain to linear."""
        return torch.pow(2.0, -y)


class ShiftLog2Quantizer(Log2Quantizer):
    """Log2 quantizer with adaptive shift."""
    
    def __init__(self, bit_width: int, channel_wise: bool, channel_num: int,
                 ema_decay: float = 0.95, rounding_cfgs: dict = None,
                 enable_distill: bool = False, scale_training: bool = False, scale_grad = True):
        """
        Args:
            enable_distill: Whether to enable distillation loss
        """
        super().__init__(bit_width, channel_wise, channel_num, ema_decay, rounding_cfgs, scale_training, scale_grad)
        self.enable_distill = enable_distill
        self.adaptive_shift = nn.Parameter(
            torch.zeros(channel_num) if channel_wise else torch.zeros(1)
        )

    def _transform_to_log(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive shift before log transform."""
        shift = self.adaptive_shift
        if self.channel_wise:
            shift = shift[None, :]
        x_shifted = (x + shift).clamp(min=self.EPS)
        return -torch.log2(x_shifted)

    def _transform_from_log(self, y: torch.Tensor) -> torch.Tensor:
        """Apply reverse shift after inverse log."""
        x = torch.pow(2.0, -y)
        shift = self.adaptive_shift
        if self.channel_wise:
            shift = shift[None, :]
        return (x - shift).clamp(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_deq = super().forward(x)
        if self.enable_distill and self.training:
            cos_sim = F.cosine_similarity(x, x_deq, dim=-1)
            self.distill_loss = 1 - cos_sim.mean()
        return x_deq

    def extra_repr(self) -> str:
        return super().extra_repr() + f", enable_distill={self.enable_distill}, adaptive_shift.shape={self.adaptive_shift.shape}"


class AdaptivePowerLawQuantizer(Log2Quantizer):
    """Power-law quantizer with adaptive parameters."""
    
    def __init__(self, bit_width: int, channel_wise: bool, channel_num: int,
                 ema_decay: float = 0.95, rounding_cfgs: dict = None,
                 log_base_ema: bool = False, enable_adaptive_shift: bool = False,
                 enable_distill: bool = False, is_post_gelu: bool = False,
                 act_clip_percentile_high: float = 100.0, act_clip_percentile_low: float = 0.0, scale_training: bool = False, scale_grad = True):
        """
        Args:
            log_base_ema: Whether to use EMA for log base
            enable_adaptive_shift: Enable adaptive shift parameter
            enable_distill: Whether to enable distillation loss
            is_post_gelu: Whether quantizer is after GELU activation
        """
        super().__init__(bit_width, channel_wise, channel_num, ema_decay, rounding_cfgs,
                         act_clip_percentile_high, act_clip_percentile_low, scale_training, scale_grad)
        self.log_base_ema = log_base_ema
        self.enable_adaptive_shift = enable_adaptive_shift
        self.enable_distill = enable_distill
        self.is_post_gelu = is_post_gelu

        self.log_base_alpha = nn.Parameter(
            torch.zeros(channel_num) if channel_wise else torch.zeros(1)
        )
        
        if enable_adaptive_shift:
            init_val = 0.17 if is_post_gelu else 0.0
            self.adaptive_shift = nn.Parameter(torch.full_like(self.log_base_alpha, init_val))

        # channel-wise shift
        # if enable_adaptive_shift:
        #     if not is_post_gelu:
        #         self.adaptive_shift = nn.Parameter(torch.full_like(self.log_base_alpha, 0.))
        #     else:
        #         self.adaptive_shift = nn.Parameter(torch.full_like(torch.zeros(channel_num), 0.17))


        if log_base_ema:
            self.register_buffer('log_base', torch.full_like(self.log_base_alpha, 2.0))

    def _transform_to_log(self, x: torch.Tensor) -> torch.Tensor:
        """Apply power-law transform to log domain."""
        if self.enable_adaptive_shift:
            shift = self.adaptive_shift
            # if self.scale_grad:
            #     shift = grad_scale(shift, self.g)
            # _shift = self.adaptive_shift
            # if self.training and self.scale_training:
            #     shift = grad_scale(_shift, self.g)
            # else:
            #     shift = _shift
            if self.channel_wise:
                shift = shift[None, :]
            x = (x + shift).clamp(min=self.EPS)
        else:
            if self.is_post_gelu:
                shift = 0.17
                x = x + shift
            x = x.clamp(min=self.EPS)
        # Compute log base
        log_base = torch.exp(self.log_base_alpha) + 1.0 # ensuring log_base is positive and initialized to 2.0
        # if self.scale_grad:
        #     log_base = grad_scale(log_base, self.g)
        # if self.training and self.scale_training:
        #     log_base = grad_scale(_log_base, self.g)
        # else:
        #     log_base = _log_base
            
        if self.log_base_ema:
            raise NotImplementedError("log_base_ema should be checked")
            if self.training:
                log_base = self.log_base * self.ema_decay + log_base * (1 - self.ema_decay)
                self.log_base.copy_(log_base.detach())  # TODO: check, seems not correct
            else:
                log_base = self.log_base
        x_log_ = -torch.log(x) / torch.log(log_base)
        
        if 0:
            x_log = grad_scale(x_log_, self.g)
        else:
            x_log = x_log_
        return x_log

    def _transform_from_log(self, y: torch.Tensor) -> torch.Tensor:
        """Transform from log domain to linear."""
        log_base = self.log_base if self.log_base_ema else torch.exp(self.log_base_alpha) + 1.0
        x_deq = torch.pow(log_base, -y)
        
        if self.enable_adaptive_shift:
            shift = self.adaptive_shift
            if self.channel_wise:
                shift = shift[None, :]
            x_deq = x_deq - shift
            
            if not self.is_post_gelu:
                x_deq = x_deq.clamp(0.0, 1.0)
        else:
            if self.is_post_gelu:
                x_deq = x_deq - 0.17
            else:
                x_deq = x_deq.clamp(0.0, 1.0)

        return x_deq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_deq = super().forward(x)
        if self.enable_distill and self.training:
            cos_sim = F.cosine_similarity(x, x_deq, dim=-1)
            self.distill_loss = 1 - cos_sim.mean()
        return x_deq

    def extra_repr(self) -> str:
        return (super().extra_repr() + 
                f", log_base_ema={self.log_base_ema}, "
                f"enable_adaptive_shift={self.enable_adaptive_shift}, "
                f"enable_distill={self.enable_distill}, "
                f"is_post_gelu={self.is_post_gelu}")

    # def adaptive_clipping(self, x: torch.Tensor):
    #     """Apply adaptive clipping to input tensor."""


class ActQ(nn.Module):
    """Learnable activation quantizer (QViT implementation)."""
    
    def __init__(self, in_features: int, nbits: int = 4, channel_wise: bool = True, **kwargs):
        """
        Args:
            in_features: Number of input features
            nbits: Number of quantization bits
            channel_wise: Whether to apply channel-wise quantization
        """
        super().__init__()
        self.nbits = nbits
        self.channel_wise = channel_wise
        self.alpha = None
        self.zero_point = None
        
        if nbits < 0:
            return

        # Initialize parameters
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.zero_point = nn.Parameter(torch.tensor([0.0]))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))

        if channel_wise:
            self.alpha = nn.Parameter(torch.Tensor(in_features))
            self.zero_point = nn.Parameter(torch.Tensor(in_features))
        
        nn.init.zeros_(self.zero_point)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with learned quantization parameters."""
        if self.alpha is None:
            return x

        # Initialization
        if self.training and self.init_state == 0:
            self.signed.data.fill_(float(x.min() < -1e-5))
            Qn = -2**(self.nbits-1) if self.signed else 0
            Qp = 2**(self.nbits-1) - 1 if self.signed else 2**self.nbits - 1
            
            alpha_init = 2 * x.abs().mean() / math.sqrt(Qp)
            zero_point_init = torch.min(x.detach()) - self.alpha.data * Qn
            self.alpha.data.copy_(alpha_init)
            self.zero_point.data.copy_(0.9 * self.zero_point.data + 0.1 * zero_point_init)
            self.init_state.fill_(1)

        # Set quantization range
        Qn = -2**(self.nbits-1) if self.signed else 0
        Qp = 2**(self.nbits-1) - 1 if self.signed else 2**self.nbits - 1
        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Quantization
        zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        alpha = grad_scale(self.alpha, g)
        zero_point = grad_scale(zero_point, g)

        # Apply quantization
        if len(x.shape) == 2:
            alpha = alpha.unsqueeze(0)
            zero_point = zero_point.unsqueeze(0)
        elif len(x.shape) == 4:
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            
        x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
        return (x - zero_point) * alpha

    def extra_repr(self) -> str:
        return f"nbits={self.nbits}, channel_wise={self.channel_wise}"


class WeightUniformQuantizer(BaseMinMaxQuantizer):
    def __init__(self, bit_width, channel_wise, channel_num, clip_percentile = 99.99, ema_decay = 0., rounding_cfgs = None, scale_training = False, scale_grad=True):
        super().__init__(bit_width, True, channel_wise, channel_num, ema_decay, rounding_cfgs, scale_training, scale_grad)
        assert 0 <= clip_percentile <= 100.0
        self.register_buffer('clip_percentile', torch.tensor(clip_percentile/100.))

    def _flatten_to_2d(self, x):
        if len(x.shape) == 2:   # linear weight, c_out * c_in
            return x.T, x.shape
        elif len(x.shape) == 4: # conv weight, c_out * c_in * k_h * k_w
            c_o, c_i, k_h, k_w = x.shape
            return x.reshape(c_o, c_i * k_h * k_w).T, x.shape
        else:
            raise ValueError
        
    def _unflatten_to_orig(self, x: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        """Restore original shape from 2D tensor."""
        if len(orig_shape) == 4:
            c_o, c_i, k_h, k_w = orig_shape
            x = x.T.reshape(c_o, c_i, k_h, k_w).contiguous()
        elif len(orig_shape) == 2:
            x = x.T.contiguous()
        else:
            raise ValueError
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with shape handling."""
        orig_dtype = x.dtype
        x_flat, orig_shape = self._flatten_to_2d(x) # [-1, c_out]
        # import pdb; pdb.set_trace()
        if self.scale_training:
            if torch.all(self.alpha == 0.): # scale not initialized
                assert not self.training
                cur_min, cur_max = self._calculate_running_minmax(x_flat.detach())
                init_alpha, init_zp = self._calculate_qparams(cur_min, cur_max)
                self.alpha.copy_(init_alpha)
                self.zero_point.copy_(init_zp)
                    
        if self.training:
            if self.scale_training: # TODO: does not impl. for scale_grad=False
                scale = self.alpha
                if self.scale_grad:
                    assert not self.g == 0.
                    scale = grad_scale(scale, self.g)
                zero_point = self.zero_point
            else:
                # we do not update weight running min/max every step, schedule
                # self._update_stats(x_flat)
                if torch.all(self.alpha == 1.):
                    min_val, max_val = self._calculate_running_minmax(x_flat)
                    alpha, zp = self._calculate_qparams(min_val, max_val)
                    self.alpha.copy_(alpha)
                    self.zero_point.copy_(zp)
                scale, zero_point = self.alpha, self.zero_point
        else:
            scale, zero_point = self.alpha, self.zero_point
        x_q = self._quantize(x_flat, scale, zero_point)
        x_deq = self._dequantize(x_q, scale, zero_point)
        x_deq = self._unflatten_to_orig(x_deq, orig_shape)
        return x_deq.to(orig_dtype)

    @torch.no_grad()
    def _calculate_running_minmax(self, x: torch.Tensor) -> tuple:
        """Compute percentile-based min/max values."""
        q = torch.tensor([1. - self.clip_percentile, self.clip_percentile], device=x.device)
        
        if self.channel_wise:
            if self.clip_percentile == 1.0:
                return torch.aminmax(x, dim=0)
            return torch.quantile(x, q=q, dim=0, keepdim=False)
        
        if self.clip_percentile == 1.0:
            return torch.aminmax(x)
        return cupy_quantile(x, q=q)
    
    def extra_repr(self) -> str:
        return super().extra_repr() + f", scale_train={self.scale_training}"
