import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from   utils.sfp_quant_utils import *



def linear_Q(q_bit, in_features, out_features):   
  class Linear_Q(nn.Linear):
    def __init__(self, in_features = in_features, out_features = out_features, bias = True):
      super(Linear_Q, self).__init__(in_features, out_features)
      self.q_bit = q_bit
      self.quantize_act = act_quantize_func(q_bit = q_bit)
      self.quantize_weight = weight_quantize_func(q_bit = q_bit)
      

    def forward(self, input, Kw, Ka):
      self.bias_q = self.bias / Kw / Ka
      self.input_q = self.quantize_act(input / Ka)
      self.weight_q = self.quantize_weight(self.weight / Kw)
      out =  F.linear(self.input_q, self.weight_q, self.bias_q) * Kw * Ka
      return out
  return Linear_Q