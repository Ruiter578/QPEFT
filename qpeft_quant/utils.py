
import torch
import cupy
import torch.nn as nn
import math
from easydict import EasyDict as edict

def cupy_quantile(a: torch.Tensor, q: torch.Tensor, dim=None, keepdim=False, method='linear'):
    # following torch.quantile
    return torch.tensor(cupy.percentile(cupy.asarray(a), cupy.asarray(q)*100, axis=dim, keepdims=keepdim,
                                        overwrite_input=False, method=method),
                        dtype=a.dtype, device=a.device)

def calculate_qminmax(bit_width: int, signed: bool):
    if signed:
        q_min = -2 ** (bit_width - 1)
        q_max = 2 ** (bit_width - 1) - 1
    else:
        q_min = 0
        q_max = 2 ** bit_width - 1

    return q_min, q_max

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x, training=True):
    y = x.round()
    if not training:
        return y
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

# --- Base class for all rounding strategies ---
class RoundingStrategy(nn.Module):
    """
    Abstract base class for different rounding strategies in quantization.
    This allows for a plug-and-play mechanism to switch between STE,
    Tanh approximation, etc.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the rounding strategy.
        
        Args:
            x (torch.Tensor): The pre-quantized, scaled tensor (e.g., values like 3.7, 4.2).
        
        Returns:
            torch.Tensor: The tensor with integer values after rounding.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    class _STE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return torch.round(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output


# --- Implementation 1: Standard STE Rounding ---
class STEQuant(RoundingStrategy):
    """
    Implements standard Straight-Through Estimator (STE) rounding.
    Forward pass: torch.round()
    Backward pass: Identity (gradient passes through unchanged).
    """
    def __init__(self):
        super().__init__()
        # This is a stateless module, but defined as a class for consistency.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._STE.apply(x)

# --- Implementation 1: Standard STE Rounding ---
class DropSTEQuant(RoundingStrategy):
    """
    Implements standard Straight-Through Estimator (STE) rounding.
    Forward pass: torch.round()
    Backward pass: Identity (gradient passes through unchanged).
    """
    def __init__(self, drop_prob = 0.):
        super().__init__()
        self.register_buffer('drop_prob', torch.tensor(drop_prob))
        # This is a stateless module, but defined as a class for consistency.

    def reset_drop_prob(self, drop_prob):
        if isinstance(drop_prob, torch.Tensor):
            self.drop_prob.copy_(drop_prob)
        else:
            self.drop_prob.copy_(torch.tensor(drop_prob))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not 0 <= self.drop_prob <= 1:
            raise ValueError("Drop probability must be between 0 and 1.") 
        if not self.training or self.drop_prob == 0.:
            return self._STE.apply(x)
        # return self._STE.apply(x)
        # mask = torch.rand_like(x) < self.drop_prob
        # x_ = torch.zeros_like(x)
        # x_[mask] = x[mask]
        # x_[~mask] = self._STE.apply(x[~mask])
        # return x_
        return torch.where(torch.rand_like(x) < self.drop_prob, x, self._STE.apply(x))

# --- Implementation 2: Progressive Tanh Approximation Rounding ---
class ProgressiveTanhQuant(RoundingStrategy):
    """
    Implements a differentiable, progressive approximation of the rounding function
    using tanh. This is designed to provide smoother gradients during the
    initial stages of training.
    
    Args:
        initial_hardness (float): The starting "hardness" of the tanh function.
                                  A lower value means a smoother function.
    """
    def __init__(self, initial_hardness: float = 6.0):
        super().__init__()
        # 'hardness' controls the steepness of the tanh function.
        # It's a buffer so it becomes part of the module's state.
        self.register_buffer('hardness', torch.tensor(initial_hardness))

    def reset_hardness(self, new_hardness: float):
        """
        Public method to update the hardness parameter during training,
        allowing for a curriculum learning approach (from smooth to sharp).
        """
        self.hardness.copy_(torch.tensor(new_hardness))
        # print(f"ProgressiveTanhQuant hardness updated to: {self.hardness.item()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the tanh-based soft rounding.
        The formula approximates round(x) by handling the fractional part smoothly.
        """
        # x_floor = torch.floor(x)
        # frac_part = x - x_floor
        
        # # This formula smoothly transitions from 0 to 1 as frac_part moves past 0.5.
        # # The steepness of this transition is controlled by `self.hardness`.
        # soft_round_offset = 0.5 * (torch.tanh(self.hardness * (frac_part - 0.5)) + 1)
        # # The final result is the integer part plus the smoothly calculated offset.
        # return x_floor + soft_round_offset
        return self.TanhQuant.apply(x, self.hardness)

    class TanhQuant(torch.autograd.Function):
        # 将hardness作为参数传入，而不是模块状态
        # 这样可以在外部进行课程学习式的调整
        @staticmethod
        def forward(ctx, x, hardness):
            ctx.save_for_backward(x, hardness)
            
            x_floor = torch.floor(x)
            frac_part = x - x_floor
            soft_round_offset = 0.5 * (torch.tanh(hardness * (frac_part - 0.5)) + 1)
            return x_floor + soft_round_offset

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None
            # x, hardness_tensor = ctx.saved_tensors
            # hardness = hardness_tensor.item()
            
            # frac_part = x - torch.floor(x)
            
            # # 这是未归一化的梯度
            # # grad_unnormalized = 0.5 * hardness * (1 - torch.tanh(hardness * (frac_part - 0.5))**2)
            
            # # 这是归一化后的梯度，其峰值为1.0
            # grad_normalized = 1 - torch.tanh(hardness * (frac_part - 0.5))**2
            
            # return grad_normalized * grad_output, None # hardness的梯度为None

def get_rounding_func(rounding_cfgs):
    if rounding_cfgs is None:
        return STEQuant()

    rounding_strategy = rounding_cfgs.get('strategy')
    if rounding_strategy == 'tanh_progressive':
        initial_hardness = rounding_cfgs.get('initial_hardness')
        return ProgressiveTanhQuant(initial_hardness=initial_hardness)
    elif rounding_strategy == 'drop_ste':
        init_drop_rate = rounding_cfgs.get('initial_drop_rate')
        return DropSTEQuant(drop_prob=init_drop_rate)
    else:
        raise ValueError(f'rounding strategy {rounding_strategy} not supported')
    