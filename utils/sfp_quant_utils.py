import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#np.set_printoptions(threshold=np.inf)

def quantize_weight(k):  
  class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): 
        if k == 32:
            out = input
        elif k == 7:
            # out = input.clone()
            N = 8   # SFP<3,3>
            sign = torch.sign(input)            
            input_abs = torch.abs(input)        
            output_abs = torch.clone(input_abs) 
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))        
            mantissa = input_abs / pow(2, exponent)         
            mantissa_q = torch.round(mantissa * N) / N            
            output_abs = torch.mul(mantissa_q, pow(2, exponent))  
            # subnormal
            output_abs[input_abs < 0.0625] = 1e-10                
            output_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            output_abs[input_abs >= 15] = 15
            out = torch.mul(sign, output_abs)

        elif k == 8:
            N = 16  # SLFP<3,4>
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs) # init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_log = torch.round(torch.log2(mantissa)*N)/N 
            output_abs = pow(2, (exponent + mantissa_log))
            # subnormal
            output_abs[input_abs < 0.0625] = 1e-10 # to avoid inf
            output_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            output_abs[input_abs > 15.32165] = 15.32165
            out = torch.mul(sign, output_abs) 
        return out

    @staticmethod
    def backward(ctx, grad_output):     # STE, do nothing in backward
      grad_input = grad_output.clone() 
      return grad_input
  return qfn().apply

def quantize_sfp34(k):  
  class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): 
        if k == 32:
            out = input
        if k == 8:
            # out = input.clone()
            N = 16   # SFP<3,4>
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs) # init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_q = torch.round(mantissa * N) / N
            output_abs = torch.mul(mantissa_q, pow(2, exponent))
            # subnormal
            output_abs[input_abs < 0.0625] = 1e-10
            output_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            output_abs[input_abs >= 15.5] = 15.5
            out = torch.mul(sign, output_abs)

        return out

    @staticmethod
    def backward(ctx, grad_output):     # STE, do nothing in backward
      grad_input = grad_output.clone() 
      return grad_input
  return qfn().apply

def quantize_slfp34(k):  
  class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): 
        if k == 32:
            out = input

        if k == 8: # SLFP<3,4>
            N = 16   # SFP<3,4>
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs) # init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_q = torch.round(mantissa * N) / N
            mantissa_log = torch.round(torch.log2(mantissa_q)*N)/N  # SLFP<3,4>, log converter
            output_abs = pow(2, (exponent + mantissa_log))
            # subnormal
            output_abs[input_abs < 0.0625] = 1e-10
            output_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            output_abs[input_abs > 15.32165] = 15.32165 # 0 111 1111
            out = torch.mul(sign, output_abs)

        return out

    @staticmethod
    def backward(ctx, grad_output):     # STE, do nothing in backward
      grad_input = grad_output.clone() 
      return grad_input
  return qfn().apply

def quantize_act(k): 
  class qfn(torch.autograd.Function): 
    @staticmethod             
    def forward(ctx, input):
        if k == 32:
            out = input

        elif k == 7:  # SFP<3,3>
            N = 8 
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs) #init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_q = torch.round(mantissa * N) / N
            output_abs = torch.mul(mantissa_q, pow(2, exponent))
            # subnormal
            output_abs[input_abs < 0.0625] = 1e-10
            output_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            output_abs[input_abs >= 15] = 15
            out = torch.mul(sign, output_abs)

        elif k == 8: # SLFP<3,4>
            N = 16   # SFP<3,4>
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs) #init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_q = torch.round(mantissa * N) / N
            mantissa_log = torch.round(torch.log2(mantissa_q)*N)/N  # SLFP<3,4>, log converter
            output_abs = pow(2, (exponent + mantissa_log))
            # subnormal
            output_abs[input_abs < 0.0625] = 1e-10
            output_abs[(input_abs >= 0.0625) & (input_abs < 0.125)] = 0.125
            # large number
            output_abs[input_abs > 15.32165] = 15.32165 # 0 111 1111
            out = torch.mul(sign, output_abs)
        return out

    @staticmethod
    def backward(ctx, grad_output): 
      grad_input = grad_output.clone()
      return grad_input
  return qfn().apply


def quantize_layerout(k):  
  class qfn(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, input): 
        if k == 32:
            out = input
        elif k <= 8:
            N = 16       #SFP<4,4>
            sign = torch.sign(input)
            input_abs = torch.abs(input)
            output_abs = torch.clone(input_abs) #init output_abs
            # normal quantization
            exponent = torch.floor(torch.log2(input_abs))
            mantissa = input_abs / pow(2, exponent)
            mantissa_q = torch.round(mantissa * N) / N
            output_abs = torch.mul(mantissa_q, pow(2, exponent))
            # subnormal
            output_abs[input_abs < 2^(-8)] = 1e-10
            output_abs[(input_abs >= 2^(-8)) & (input_abs < 2^(-7))] = 2^(-7)
            # large number
            output_abs[input_abs >= 248] = 248 # 
            out = torch.mul(sign, output_abs)
        return out

    @staticmethod
    def backward(ctx, grad_output):     
      grad_input = grad_output.clone() 
      return grad_input
  return qfn().apply

class weight_quantize_func(nn.Module):    
  def __init__(self, q_bit):
    super(weight_quantize_func, self).__init__()
    assert q_bit <= 8 or q_bit == 32
    self.q_bit = q_bit      
    self.quantize = quantize_weight(k=q_bit) 

  def forward(self, x):
    if self.q_bit == 32:
      weight_q = x
    elif self.q_bit == 8 or self.q_bit == 7:
      weight_q = self.quantize(x)  
    return weight_q

class act_quantize_func(nn.Module): 
  def __init__(self, q_bit):
    super(act_quantize_func, self).__init__()
    assert q_bit <= 8 or q_bit == 32
    self.q_bit = q_bit                  
    self.quantize = quantize_act(k=q_bit) 

  def forward(self, x):
    if self.q_bit == 32:
      act_q = x
    elif self.q_bit == 8 or self.q_bit == 7:
      act_q = self.quantize(x)  
    return act_q

class layerout_quantize_func(nn.Module): 
  def __init__(self, q_bit):
    super(layerout_quantize_func, self).__init__()
    assert q_bit <= 8 or q_bit == 32
    self.q_bit = q_bit 
    self.quantize = quantize_layerout(k = q_bit) 

  def forward(self, x):
    if self.q_bit == 32:
      out_q = x
    elif self.q_bit == 8 or self.q_bit == 7:
      out_q = self.quantize(x)  
    return out_q
