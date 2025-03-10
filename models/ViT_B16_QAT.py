import copy, logging
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from os.path import join
from scipy import ndimage
from utils.linear_utils import *
""" + get_scale_factor"""
from utils.get_scale_factor_utils import *

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights):
    if weights.ndim == 4:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class SelfAttention(nn.Module):
    """ + scaling factor definition """
    def __init__(self, config, vis, qbits, kmha_i, kmha_q, kmha_k, kmha_v, kmha_o, kmha_wo):
        super(SelfAttention, self).__init__()

        self.vis = vis
        self.num_key_value_head = config.transformer.num_key_value_head
        self.head_num = config.transformer.head_num
        self.head_dim = int(config.hidden_size // config.transformer.head_num)
        self.head_dim_sum = self.head_num * self.head_dim

        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(config.transformer.attention_dropout)
        self.dropout = nn.Dropout(config.transformer.dropout)
        self.scale = self.head_dim ** -0.5

        """ + list to tensor"""
        self.kmha_i  =   torch.tensor(kmha_i)
        self.kmha_q  =   torch.tensor(kmha_q)
        self.kmha_k  =   torch.tensor(kmha_k)
        self.kmha_v  =   torch.tensor(kmha_v)
        self.kmha_o  =   torch.tensor(kmha_o)
        self.kmha_wo =   torch.tensor(kmha_wo)


        """ - linear """
        ''' 
        self.Wk = nn.Linear(config.hidden_size,
                            self.head_num * self.head_dim,
                            bias=config.transformer.qkv_bias)

        self.Wq = nn.Linear(config.hidden_size,
                            self.head_num * self.head_dim,
                            bias=config.transformer.qkv_bias)
        
        self.Wv = nn.Linear(config.hidden_size,
                            self.head_num * self.head_dim,
                            bias=config.transformer.qkv_bias)
        '''

        """ + linear quantization """
        Linear_Wkqv = linear_Q(q_bit = qbits, in_features = config.hidden_size, out_features = self.head_num * self.head_dim) 

        
        """ + wk, wq and wv quantization """
        self.Wk = Linear_Wkqv()
        self.Wq = Linear_Wkqv()
        self.Wv = Linear_Wkqv()

        """ - linear """
        # self.out = nn.Linear(config.hidden_size, config.hidden_size)

        """ + linear quantization"""
        Linear_Out = linear_Q(q_bit = qbits, in_features = config.hidden_size, out_features = config.hidden_size)
        self.out = Linear_Out()


        """ + Q, K and V quantization"""
        self.quantize_act = act_quantize_func(q_bit = qbits)
        self.quantize_weight = weight_quantize_func(q_bit = qbits)


        if (self.head_dim * self.head_num) != config.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.head_num}).")

    def forward(self, hidden_states):

        """ + Ka, Kw """
        Q = self.Wq(hidden_states, Ka = self.kmha_i, Kw = self.kmha_q)
        K = self.Wk(hidden_states, Ka = self.kmha_i, Kw = self.kmha_k)
        V = self.Wv(hidden_states, Ka = self.kmha_i, Kw = self.kmha_v)
        # adjust shapes to be (batch, head_num, seq_len, head_dim)
        batch_size, seq_len, _ = hidden_states.size()
        Q = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_key_value_head, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_key_value_head, self.head_dim).transpose(1, 2)

        # """ + Q, K and V quantization """
        # Q = (Q * self.KW_q * self.KA) / self.KQ
        # K = (K * self.KW_k * self.KA) / self.KK
        # V = (V * self.KW_v * self.KA) / self.KV

        # Q = self.quantize_act(Q) 
        # K = self.quantize_act(K) 
        # V = self.quantize_act(V) 

        # """ + Q, K and V reverse scaling quantization """
        # Q = Q * self.kmha_i * self.kmha_q
        # K = K * self.kmha_i * self.kmha_k
        # V = V * self.kmha_i * self.kmha_v

        # calculate attention scores and output
        weights = torch.matmul(Q, K.transpose(-1, -2))
        weights = self.softmax(weights * self.scale)
        vis_weights = weights if self.vis else None
        weights = self.attention_dropout(weights)
        output = torch.matmul(weights, V)

        # reshape to be (batch, seq_len, hidden_size)
        output = output.permute(0, 2, 1, 3).contiguous()
        output_shape = output.size()[:-2] + (self.head_dim_sum,)
        output = output.view(*output_shape)
        
        # output projection
        """ + Ka, Kw """
        output = self.out(output, Ka = self.kmha_o, Kw = self.kmha_wo)
        output = self.dropout(output)


        
        return output, vis_weights


class MLP(nn.Module):
    def __init__(self, config, qbits, kmlp_ifc1, kmlp_wfc1, kmlp_ifc2, kmlp_wfc2):
        super(MLP, self).__init__()
        self.input_dim = config.hidden_size
        self.hidden_dim = config.transformer.mlp_dim
        self.dropout_rate = config.transformer.dropout

        """ + list to tensor"""
        self.kmlp_ifc1  =   torch.tensor(kmlp_ifc1)
        self.kmlp_wfc1  =   torch.tensor(kmlp_wfc1)

        self.kmlp_ifc2  =   torch.tensor(kmlp_ifc2)
        self.kmlp_wfc2  =   torch.tensor(kmlp_wfc2)

        """ - nn.Linear()"""
        # self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        # self.fc2 = nn.Linear(self.hidden_dim, self.input_dim)

        """ + MPL linear quantization """
        Linear_MPL_fc1 = linear_Q(q_bit = qbits, in_features = self.input_dim,  out_features = self.hidden_dim)
        Linear_MPL_fc2 = linear_Q(q_bit = qbits, in_features = self.hidden_dim, out_features = self.input_dim)
        self.fc1 = Linear_MPL_fc1()
        self.fc2 = Linear_MPL_fc2()


        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = F.gelu

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, hidden_state):
        hidden_state = self.fc1(hidden_state, Ka = self.kmlp_ifc1, Kw = self.kmlp_wfc1)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.fc2(hidden_state, Ka = self.kmlp_ifc2, Kw = self.kmlp_wfc2)
        output = self.dropout(hidden_state)
        return output


class EncoderBlock(nn.Module):
    """ +   qbits
        +   kmha_i, kmha_q, kmha_k, kmha_v, kmha_o, kmha_wo
        +   kmlp_ifc1, kmlp_wfc1, kmlp_ifc2, kmlp_wfc2
    """
    def __init__(self, config, vis, 
                 qbits, 
                 kmha_i, kmha_q, kmha_k, kmha_v, kmha_o, kmha_wo,
                 kmlp_ifc1, kmlp_wfc1, kmlp_ifc2, kmlp_wfc2):
        super(EncoderBlock, self).__init__()
        self.vis = vis
        self.hidden_size = config.hidden_size
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        """ + qbits, kmha_i, kmha_q, kmha_k, kmha_v, kmha_o, kmha_wo """
        self.selfattention = SelfAttention(config, self.vis, qbits, kmha_i, kmha_q, kmha_k, kmha_v, kmha_o, kmha_wo)
        """ + qbits, kmlp_ifc1, kmlp_wfc1, kmlp_ifc2, kmlp_wfc2 """
        self.mlp = MLP(config, qbits, kmlp_ifc1, kmlp_wfc1, kmlp_ifc2, kmlp_wfc2)

    def forward(self, embed_state):
        skip_state = embed_state
        hidden_state = self.attn_norm(embed_state)
        hidden_state, weights = self.selfattention(hidden_state)
        hidden_state += skip_state
        
        skip_state = hidden_state
        hidden_state = self.mlp_norm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        output = hidden_state + skip_state
        return output, weights
    
    def load_weights(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(
                weights[join(ROOT,ATTENTION_Q, "kernel").replace("\\","/")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(
                weights[join(ROOT, ATTENTION_K, "kernel").replace("\\","/")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(
                weights[join(ROOT, ATTENTION_V, "kernel").replace("\\","/")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(
                weights[join(ROOT, ATTENTION_OUT, "kernel").replace("\\","/")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(
                weights[join(ROOT, ATTENTION_Q, "bias").replace("\\","/")]).view(-1)
            key_bias = np2th(
                weights[join(ROOT, ATTENTION_K, "bias").replace("\\","/")]).view(-1)
            value_bias = np2th(
                weights[join(ROOT, ATTENTION_V, "bias").replace("\\","/")]).view(-1)
            out_bias = np2th(
                weights[join(ROOT, ATTENTION_OUT, "bias").replace("\\","/")]).view(-1)

            self.selfattention.Wq.weight.copy_(query_weight)
            self.selfattention.Wk.weight.copy_(key_weight)
            self.selfattention.Wv.weight.copy_(value_weight)
            self.selfattention.out.weight.copy_(out_weight)
            
            self.selfattention.Wq.bias.copy_(query_bias)
            self.selfattention.Wk.bias.copy_(key_bias)
            self.selfattention.Wv.bias.copy_(value_bias)
            self.selfattention.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(
                weights[join(ROOT, FC_0, "kernel").replace("\\","/")]).t()
            mlp_weight_1 = np2th(
                weights[join(ROOT, FC_1, "kernel").replace("\\","/")]).t()
            mlp_bias_0 = np2th(
                weights[join(ROOT, FC_0, "bias").replace("\\","/")]).t()
            mlp_bias_1 = np2th(
                weights[join(ROOT, FC_1, "bias").replace("\\","/")]).t()

            self.mlp.fc1.weight.copy_(mlp_weight_0)
            self.mlp.fc2.weight.copy_(mlp_weight_1)
            self.mlp.fc1.bias.copy_(mlp_bias_0)
            self.mlp.fc2.bias.copy_(mlp_bias_1)

            self.attn_norm.weight.copy_(np2th(
                weights[join(ROOT, ATTENTION_NORM, "scale").replace("\\","/")]))
            self.attn_norm.bias.copy_(np2th(
                weights[join(ROOT, ATTENTION_NORM, "bias").replace("\\","/")]))
            self.mlp_norm.weight.copy_(np2th(
                weights[join(ROOT, MLP_NORM, "scale").replace("\\","/")]))
            self.mlp_norm.bias.copy_(np2th(
                weights[join(ROOT, MLP_NORM, "bias").replace("\\","/")]))


class TransformerEndocer(nn.Module):
    def __init__(self, config, vis, qbits):
        super(TransformerEndocer, self).__init__()
        self.vis = vis
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.layer_num = config.transformer.layer_num
        self.encoder_layer = nn.ModuleList()

        """ + Kmha_i, Kmha_q, Kmha_k, Kmha_v, Kmha_o, Kmha_wo
            + Kmlp_ifc1, Kmlp_wfc1, Kmlp_ifc2, Kmlp_wfc2
        """
        folder_path = "./scale_factor/"

        """ + fp32 or slfp"""
        if qbits == 32:
            """ + mha """
            Kmha_i  = [1] * self.layer_num
            Kmha_q  = [1] * self.layer_num
            Kmha_k  = [1] * self.layer_num
            Kmha_v  = [1] * self.layer_num
            Kmha_o  = [1] * self.layer_num
            Kmha_wo = [1] * self.layer_num

            """ + mlp """
            Kmlp_ifc1 = [1] * self.layer_num
            Kmlp_wfc1 = [1] * self.layer_num
            Kmlp_ifc2 = [1] * self.layer_num
            Kmlp_wfc2 = [1] * self.layer_num


        else :
            """ + mha """
            MAXIMUM_MAGNITUDE = 15.5
            Kmha_i, Kmha_k, Kmha_q, Kmha_v, Kmha_o, Kmha_wo = acquire_mha_layer_scale_factor_txt(folder_path)
            Kmha_i  = np.array(Kmha_i) / MAXIMUM_MAGNITUDE
            Kmha_k  = np.array(Kmha_k) / MAXIMUM_MAGNITUDE
            Kmha_q  = np.array(Kmha_q) / MAXIMUM_MAGNITUDE
            Kmha_v  = np.array(Kmha_v) / MAXIMUM_MAGNITUDE
            Kmha_o  = np.array(Kmha_o) / MAXIMUM_MAGNITUDE
            Kmha_wo = np.array(Kmha_wo)/ MAXIMUM_MAGNITUDE

            """ + mlp """
            Kmlp_ifc1, Kmlp_wfc1, Kmlp_ifc2, Kmlp_wfc2 = acquire_mlp_layer_scale_factor_txt(folder_path)
            Kmlp_ifc1  = np.array(Kmlp_ifc1) / MAXIMUM_MAGNITUDE
            Kmlp_wfc1  = np.array(Kmlp_wfc1) / MAXIMUM_MAGNITUDE
            Kmlp_ifc2  = np.array(Kmlp_ifc2) / MAXIMUM_MAGNITUDE
            Kmlp_wfc2  = np.array(Kmlp_wfc2) / MAXIMUM_MAGNITUDE


        for index , _ in enumerate(range(self.layer_num)):
            """ +   qbits
                +   kmha_i, kmha_q, kmha_k, kmha_v, kmha_o, kmha_wo
                +   kmlp_ifc1, kmlp_wfc1, kmlp_ifc2, kmlp_wfc2
            """
            layer = EncoderBlock(config, self.vis, 
                                 qbits,
                                 kmha_i = Kmha_i[index], kmha_q = Kmha_q[index], kmha_k = Kmha_k[index], kmha_v = Kmha_v[index], kmha_o = Kmha_o[index], kmha_wo = Kmha_wo[index],
                                 kmlp_ifc1 = Kmlp_ifc1[index], kmlp_wfc1 = Kmlp_wfc1[index], kmlp_ifc2 = Kmlp_ifc2[index], kmlp_wfc2 = Kmlp_wfc2[index])
            self.encoder_layer.append(copy.deepcopy(layer))

    def forward(self, hidden_state):
        attn_weights = []
        for layer in self.encoder_layer:
            hidden_state, weight = layer(hidden_state)
            if self.vis:
                attn_weights.append(weight)
        output = self.encoder_norm(hidden_state)



        return output, attn_weights


class Embedding(nn.Module):
    def __init__(self, config, img_size, img_channels=3):
        super(Embedding, self).__init__()

        # transfer an integer into (img_size, img_size)
        self.img_size = _pair(img_size)
        self.patch_size = _pair(config.patch['size'])
        self.patch_num = (img_size[0]//self.patch_size[0]) * (img_size[1]//self.patch_size[1])
        self.dropout = nn.Dropout(config.transformer.dropout)

        # patch embedding by filters
        self.patch_embed = nn.Conv2d(img_channels,
                                     config.hidden_size,
                                     kernel_size=self.patch_size,
                                     stride=self.patch_size,
                                     bias=True)
        
        # setting positional encoding and <cls> as parameters for learning
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.patch_num+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, img):
        batch_num = img.shape[0]
        cls_token = self.cls_token.expand(batch_num, -1, -1)
        features = self.patch_embed(img)

        # flatten all dimesions accpet for the first(batch)
        features = features.flatten(2)
        features = features.transpose(-1, -2)
        features = torch.cat((cls_token, features), dim=1)

        # positional encoding
        features += self.pos_encoding
        features = self.dropout(features)

        return features


class VisionTransformer(nn.Module):
    def __init__(self, config, load_head=True, vis=False, qbits = 32, pre_reference = False):
        super(VisionTransformer, self).__init__()
        self.load_head = load_head
        self.vis = vis
        self.img_size2d = config.input_size[1:]
        self.img_channel = config.input_size[0]
        self.embedding_layer = Embedding(config, self.img_size2d, self.img_channel)

        """ + pre_reference """
        self.pre_reference = pre_reference

        """ + linear quantization """
        Linear_MLP_Head = linear_Q(q_bit = qbits, in_features = config.hidden_size, out_features = config.num_classes) 
        
        """ + qbits"""
        self.feature_layer = TransformerEndocer(config, self.vis, qbits)


        # self.mlp_head = nn.Linear(config.hidden_size, config.num_classes)
        self.mlp_head = Linear_MLP_Head()

        """ Kmlp_head_i, Kmlp_head_w """
        folder_path = "./scale_factor/"

        """ + fp32 or slfp"""
        if qbits == 32:
            self.kmlp_head_i  = 1
            self.kmlp_head_w  = 1
        else:
            MAXIMUM_MAGNITUDE = 15.5
            self.kmlp_head_i  = 0.9120541214942932 / MAXIMUM_MAGNITUDE
            self.kmlp_head_w  = 0.3339173197746277 / MAXIMUM_MAGNITUDE


        """ + weights and inputs and outputs for each layer"""
        """ + mha """
        self.layer_mha_input   =  {}
        self.layer_mha_wk      =  {}
        self.layer_mha_wq      =  {}
        self.layer_mha_wv      =  {}
        self.layer_mha_output  =  {}
        self.layer_mha_wo      =  {} 

        """ + mlp """
        self.layer_mlp_ifc1    =  {}
        self.layer_mlp_wfc1    =  {}
        self.layer_mlp_ifc2    =  {}
        self.layer_mlp_wfc2    =  {} 

        """ + mlp head"""
        self.mlp_head_i        =  {}
        self.mlp_head_w        =  {}


    
    def load_weights(self, weights):
        with torch.no_grad():
            if not self.load_head:
                nn.init.xavier_uniform_(self.mlp_head.weight)
                nn.init.zeros_(self.mlp_head.bias)
            else:
                self.mlp_head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.mlp_head.bias.copy_(np2th(weights["head/bias"]).t())

            # load patch embedding filter weights
            self.embedding_layer.patch_embed.weight.copy_(np2th(weights["embedding/kernel"]))
            self.embedding_layer.patch_embed.bias.copy_(np2th(weights["embedding/bias"]))
            # load <cls> weights
            self.embedding_layer.cls_token.copy_(np2th(weights["cls"]))
            # load encoder layernorm weights
            self.feature_layer.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.feature_layer.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            # load positional embedding weights
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_origin = self.embedding_layer.pos_encoding
            if posemb.size() == posemb_origin.size():
                self.embedding_layer.pos_encoding.copy_(posemb)
            else:
                logger = logging.getLogger(__name__)
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_origin.size()))
                ntok_new = posemb_origin.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # nn.Module.named_children() → Iterator[Tuple[str, nn.Module]]
            for banme, block in self.feature_layer.named_children():
                for uname, unit in block.named_children():
                    # get each encoder block from block(12 layers)
                    unit.load_weights(weights, n_block=uname)

        
    def forward(self, input_ids):
        embed_feature = self.embedding_layer(input_ids)
        features, attn_weights = self.feature_layer(embed_feature)
        
        # fc with first feature of Transformer output
        logits = self.mlp_head(features[:, 0], self.kmlp_head_i, self.kmlp_head_w)

        
        """ + weights and inputs and outputs for each layer"""
        if self.pre_reference == True:
            """ layer0 """
            # mha
            self.layer_mha_input [0]     = self.feature_layer.encoder_layer._modules['0'].selfattention.Wk.input_q
            self.layer_mha_wk    [0]     = self.feature_layer.encoder_layer._modules['0'].selfattention.Wk.weight_q
            self.layer_mha_wq    [0]     = self.feature_layer.encoder_layer._modules['0'].selfattention.Wq.weight_q
            self.layer_mha_wv    [0]     = self.feature_layer.encoder_layer._modules['0'].selfattention.Wv.weight_q
            self.layer_mha_output[0]     = self.feature_layer.encoder_layer._modules['0'].selfattention.out.input_q
            self.layer_mha_wo    [0]     = self.feature_layer.encoder_layer._modules['0'].selfattention.out.weight_q
            # mlp 
            self.layer_mlp_ifc1  [0]     =  self.feature_layer.encoder_layer._modules['0'].mlp.fc1.input_q
            self.layer_mlp_wfc1  [0]     =  self.feature_layer.encoder_layer._modules['0'].mlp.fc1.weight_q
            self.layer_mlp_ifc2  [0]     =  self.feature_layer.encoder_layer._modules['0'].mlp.fc2.input_q
            self.layer_mlp_wfc2  [0]     =  self.feature_layer.encoder_layer._modules['0'].mlp.fc2.weight_q

            """ layer1 """
            # mha
            self.layer_mha_input [1]     = self.feature_layer.encoder_layer._modules['1'].selfattention.Wk.input_q
            self.layer_mha_wk    [1]     = self.feature_layer.encoder_layer._modules['1'].selfattention.Wk.weight_q
            self.layer_mha_wq    [1]     = self.feature_layer.encoder_layer._modules['1'].selfattention.Wq.weight_q
            self.layer_mha_wv    [1]     = self.feature_layer.encoder_layer._modules['1'].selfattention.Wv.weight_q
            self.layer_mha_output[1]     = self.feature_layer.encoder_layer._modules['1'].selfattention.out.input_q
            self.layer_mha_wo    [1]     = self.feature_layer.encoder_layer._modules['1'].selfattention.out.weight_q
            # mlp 
            self.layer_mlp_ifc1  [1]     =  self.feature_layer.encoder_layer._modules['1'].mlp.fc1.input_q
            self.layer_mlp_wfc1  [1]     =  self.feature_layer.encoder_layer._modules['1'].mlp.fc1.weight_q
            self.layer_mlp_ifc2  [1]     =  self.feature_layer.encoder_layer._modules['1'].mlp.fc2.input_q
            self.layer_mlp_wfc2  [1]     =  self.feature_layer.encoder_layer._modules['1'].mlp.fc2.weight_q

            """ layer2 """
            # mha
            self.layer_mha_input [2]     = self.feature_layer.encoder_layer._modules['2'].selfattention.Wk.input_q
            self.layer_mha_wk    [2]     = self.feature_layer.encoder_layer._modules['2'].selfattention.Wk.weight_q
            self.layer_mha_wq    [2]     = self.feature_layer.encoder_layer._modules['2'].selfattention.Wq.weight_q
            self.layer_mha_wv    [2]     = self.feature_layer.encoder_layer._modules['2'].selfattention.Wv.weight_q
            self.layer_mha_output[2]     = self.feature_layer.encoder_layer._modules['2'].selfattention.out.input_q
            self.layer_mha_wo    [2]     = self.feature_layer.encoder_layer._modules['2'].selfattention.out.weight_q
            # mlp 
            self.layer_mlp_ifc1  [2]     =  self.feature_layer.encoder_layer._modules['2'].mlp.fc1.input_q
            self.layer_mlp_wfc1  [2]     =  self.feature_layer.encoder_layer._modules['2'].mlp.fc1.weight_q
            self.layer_mlp_ifc2  [2]     =  self.feature_layer.encoder_layer._modules['2'].mlp.fc2.input_q
            self.layer_mlp_wfc2  [2]     =  self.feature_layer.encoder_layer._modules['2'].mlp.fc2.weight_q

            """ layer3 """
            # mha
            self.layer_mha_input [3]     = self.feature_layer.encoder_layer._modules['3'].selfattention.Wk.input_q
            self.layer_mha_wk    [3]     = self.feature_layer.encoder_layer._modules['3'].selfattention.Wk.weight_q
            self.layer_mha_wq    [3]     = self.feature_layer.encoder_layer._modules['3'].selfattention.Wq.weight_q
            self.layer_mha_wv    [3]     = self.feature_layer.encoder_layer._modules['3'].selfattention.Wv.weight_q
            self.layer_mha_output[3]     = self.feature_layer.encoder_layer._modules['3'].selfattention.out.input_q
            self.layer_mha_wo    [3]     = self.feature_layer.encoder_layer._modules['3'].selfattention.out.weight_q
            # mlp 
            self.layer_mlp_ifc1  [3]     =  self.feature_layer.encoder_layer._modules['3'].mlp.fc1.input_q
            self.layer_mlp_wfc1  [3]     =  self.feature_layer.encoder_layer._modules['3'].mlp.fc1.weight_q
            self.layer_mlp_ifc2  [3]     =  self.feature_layer.encoder_layer._modules['3'].mlp.fc2.input_q
            self.layer_mlp_wfc2  [3]     =  self.feature_layer.encoder_layer._modules['3'].mlp.fc2.weight_q

            """ layer4 """
            # mha
            self.layer_mha_input [4]     = self.feature_layer.encoder_layer._modules['4'].selfattention.Wk.input_q
            self.layer_mha_wk    [4]     = self.feature_layer.encoder_layer._modules['4'].selfattention.Wk.weight_q
            self.layer_mha_wq    [4]     = self.feature_layer.encoder_layer._modules['4'].selfattention.Wq.weight_q
            self.layer_mha_wv    [4]     = self.feature_layer.encoder_layer._modules['4'].selfattention.Wv.weight_q
            self.layer_mha_output[4]     = self.feature_layer.encoder_layer._modules['4'].selfattention.out.input_q
            self.layer_mha_wo    [4]     = self.feature_layer.encoder_layer._modules['4'].selfattention.out.weight_q
            # mlp 
            self.layer_mlp_ifc1  [4]     =  self.feature_layer.encoder_layer._modules['4'].mlp.fc1.input_q
            self.layer_mlp_wfc1  [4]     =  self.feature_layer.encoder_layer._modules['4'].mlp.fc1.weight_q
            self.layer_mlp_ifc2  [4]     =  self.feature_layer.encoder_layer._modules['4'].mlp.fc2.input_q
            self.layer_mlp_wfc2  [4]     =  self.feature_layer.encoder_layer._modules['4'].mlp.fc2.weight_q

            """ layer5 """
            # mha
            self.layer_mha_input [5]     = self.feature_layer.encoder_layer._modules['5'].selfattention.Wk.input_q
            self.layer_mha_wk    [5]     = self.feature_layer.encoder_layer._modules['5'].selfattention.Wk.weight_q
            self.layer_mha_wq    [5]     = self.feature_layer.encoder_layer._modules['5'].selfattention.Wq.weight_q
            self.layer_mha_wv    [5]     = self.feature_layer.encoder_layer._modules['5'].selfattention.Wv.weight_q
            self.layer_mha_output[5]     = self.feature_layer.encoder_layer._modules['5'].selfattention.out.input_q
            self.layer_mha_wo    [5]     = self.feature_layer.encoder_layer._modules['5'].selfattention.out.weight_q
            # mlp 
            self.layer_mlp_ifc1  [5]     =  self.feature_layer.encoder_layer._modules['5'].mlp.fc1.input_q
            self.layer_mlp_wfc1  [5]     =  self.feature_layer.encoder_layer._modules['5'].mlp.fc1.weight_q
            self.layer_mlp_ifc2  [5]     =  self.feature_layer.encoder_layer._modules['5'].mlp.fc2.input_q
            self.layer_mlp_wfc2  [5]     =  self.feature_layer.encoder_layer._modules['5'].mlp.fc2.weight_q

            """ layer6 """
            # mha
            self.layer_mha_input [6]     = self.feature_layer.encoder_layer._modules['6'].selfattention.Wk.input_q
            self.layer_mha_wk    [6]     = self.feature_layer.encoder_layer._modules['6'].selfattention.Wk.weight_q
            self.layer_mha_wq    [6]     = self.feature_layer.encoder_layer._modules['6'].selfattention.Wq.weight_q
            self.layer_mha_wv    [6]     = self.feature_layer.encoder_layer._modules['6'].selfattention.Wv.weight_q
            self.layer_mha_output[6]     = self.feature_layer.encoder_layer._modules['6'].selfattention.out.input_q
            self.layer_mha_wo    [6]     = self.feature_layer.encoder_layer._modules['6'].selfattention.out.weight_q
            # mlp 
            self.layer_mlp_ifc1  [6]     =  self.feature_layer.encoder_layer._modules['6'].mlp.fc1.input_q
            self.layer_mlp_wfc1  [6]     =  self.feature_layer.encoder_layer._modules['6'].mlp.fc1.weight_q
            self.layer_mlp_ifc2  [6]     =  self.feature_layer.encoder_layer._modules['6'].mlp.fc2.input_q
            self.layer_mlp_wfc2  [6]     =  self.feature_layer.encoder_layer._modules['6'].mlp.fc2.weight_q

            """ layer7 """
            # mha
            self.layer_mha_input [7]     = self.feature_layer.encoder_layer._modules['7'].selfattention.Wk.input_q
            self.layer_mha_wk    [7]     = self.feature_layer.encoder_layer._modules['7'].selfattention.Wk.weight_q
            self.layer_mha_wq    [7]     = self.feature_layer.encoder_layer._modules['7'].selfattention.Wq.weight_q
            self.layer_mha_wv    [7]     = self.feature_layer.encoder_layer._modules['7'].selfattention.Wv.weight_q
            self.layer_mha_output[7]     = self.feature_layer.encoder_layer._modules['7'].selfattention.out.input_q
            self.layer_mha_wo    [7]     = self.feature_layer.encoder_layer._modules['7'].selfattention.out.weight_q
            # mlp 
            self.layer_mlp_ifc1  [7]     =  self.feature_layer.encoder_layer._modules['7'].mlp.fc1.input_q
            self.layer_mlp_wfc1  [7]     =  self.feature_layer.encoder_layer._modules['7'].mlp.fc1.weight_q
            self.layer_mlp_ifc2  [7]     =  self.feature_layer.encoder_layer._modules['7'].mlp.fc2.input_q
            self.layer_mlp_wfc2  [7]     =  self.feature_layer.encoder_layer._modules['7'].mlp.fc2.weight_q

            """ layer8 """
            # mha
            self.layer_mha_input [8]     = self.feature_layer.encoder_layer._modules['8'].selfattention.Wk.input_q
            self.layer_mha_wk    [8]     = self.feature_layer.encoder_layer._modules['8'].selfattention.Wk.weight_q
            self.layer_mha_wq    [8]     = self.feature_layer.encoder_layer._modules['8'].selfattention.Wq.weight_q
            self.layer_mha_wv    [8]     = self.feature_layer.encoder_layer._modules['8'].selfattention.Wv.weight_q
            self.layer_mha_output[8]     = self.feature_layer.encoder_layer._modules['8'].selfattention.out.input_q
            self.layer_mha_wo    [8]     = self.feature_layer.encoder_layer._modules['8'].selfattention.out.weight_q
            # mlp 
            self.layer_mlp_ifc1  [8]     =  self.feature_layer.encoder_layer._modules['8'].mlp.fc1.input_q
            self.layer_mlp_wfc1  [8]     =  self.feature_layer.encoder_layer._modules['8'].mlp.fc1.weight_q
            self.layer_mlp_ifc2  [8]     =  self.feature_layer.encoder_layer._modules['8'].mlp.fc2.input_q
            self.layer_mlp_wfc2  [8]     =  self.feature_layer.encoder_layer._modules['8'].mlp.fc2.weight_q

            """ layer9 """
            # mha
            self.layer_mha_input [9]     = self.feature_layer.encoder_layer._modules['9'].selfattention.Wk.input_q
            self.layer_mha_wk    [9]     = self.feature_layer.encoder_layer._modules['9'].selfattention.Wk.weight_q
            self.layer_mha_wq    [9]     = self.feature_layer.encoder_layer._modules['9'].selfattention.Wq.weight_q
            self.layer_mha_wv    [9]     = self.feature_layer.encoder_layer._modules['9'].selfattention.Wv.weight_q
            self.layer_mha_output[9]     = self.feature_layer.encoder_layer._modules['9'].selfattention.out.input_q
            self.layer_mha_wo    [9]     = self.feature_layer.encoder_layer._modules['9'].selfattention.out.weight_q
            # mlp 
            self.layer_mlp_ifc1  [9]     =  self.feature_layer.encoder_layer._modules['9'].mlp.fc1.input_q
            self.layer_mlp_wfc1  [9]     =  self.feature_layer.encoder_layer._modules['9'].mlp.fc1.weight_q
            self.layer_mlp_ifc2  [9]     =  self.feature_layer.encoder_layer._modules['9'].mlp.fc2.input_q
            self.layer_mlp_wfc2  [9]     =  self.feature_layer.encoder_layer._modules['9'].mlp.fc2.weight_q

            """ layer10 """
            # mha
            self.layer_mha_input [10]     = self.feature_layer.encoder_layer._modules['10'].selfattention.Wk.input_q
            self.layer_mha_wk    [10]     = self.feature_layer.encoder_layer._modules['10'].selfattention.Wk.weight_q
            self.layer_mha_wq    [10]     = self.feature_layer.encoder_layer._modules['10'].selfattention.Wq.weight_q
            self.layer_mha_wv    [10]     = self.feature_layer.encoder_layer._modules['10'].selfattention.Wv.weight_q
            self.layer_mha_output[10]     = self.feature_layer.encoder_layer._modules['10'].selfattention.out.input_q
            self.layer_mha_wo    [10]     = self.feature_layer.encoder_layer._modules['10'].selfattention.out.weight_q
            # mlp 
            self.layer_mlp_ifc1  [10]     =  self.feature_layer.encoder_layer._modules['10'].mlp.fc1.input_q
            self.layer_mlp_wfc1  [10]     =  self.feature_layer.encoder_layer._modules['10'].mlp.fc1.weight_q
            self.layer_mlp_ifc2  [10]     =  self.feature_layer.encoder_layer._modules['10'].mlp.fc2.input_q
            self.layer_mlp_wfc2  [10]     =  self.feature_layer.encoder_layer._modules['10'].mlp.fc2.weight_q

            """ layer11 """
            # mha
            self.layer_mha_input [11]     = self.feature_layer.encoder_layer._modules['11'].selfattention.Wk.input_q
            self.layer_mha_wk    [11]     = self.feature_layer.encoder_layer._modules['11'].selfattention.Wk.weight_q
            self.layer_mha_wq    [11]     = self.feature_layer.encoder_layer._modules['11'].selfattention.Wq.weight_q
            self.layer_mha_wv    [11]     = self.feature_layer.encoder_layer._modules['11'].selfattention.Wv.weight_q
            self.layer_mha_output[11]     = self.feature_layer.encoder_layer._modules['11'].selfattention.out.input_q
            self.layer_mha_wo    [11]     = self.feature_layer.encoder_layer._modules['11'].selfattention.out.weight_q
            # mlp 
            self.layer_mlp_ifc1  [11]     =  self.feature_layer.encoder_layer._modules['11'].mlp.fc1.input_q
            self.layer_mlp_wfc1  [11]     =  self.feature_layer.encoder_layer._modules['11'].mlp.fc1.weight_q
            self.layer_mlp_ifc2  [11]     =  self.feature_layer.encoder_layer._modules['11'].mlp.fc2.input_q
            self.layer_mlp_wfc2  [11]     =  self.feature_layer.encoder_layer._modules['11'].mlp.fc2.weight_q

            """ mlp head"""
            self.mlp_head_i = self.mlp_head.input_q
            self.mlp_head_w = self.mlp_head.weight_q


        return logits, attn_weights

    """ + get layer_mha_data"""
    def get_layer_mha_data(self):
        return self.layer_mha_input, self.layer_mha_wk, self.layer_mha_wq, self.layer_mha_wv, self.layer_mha_output, self.layer_mha_wo
    
    """ + get layer_mlp_data"""
    def get_layer_mlp_data(self):
        return self.layer_mlp_ifc1, self.layer_mlp_wfc1, self.layer_mlp_ifc2, self.layer_mlp_wfc2

    """ + get mlp_head_data"""
    def get_mlp_head_data(self):
        return  self.mlp_head_i, self.mlp_head_w
    

    
    """ + reset layer_mha_data"""
    def reset_layer_mha_data(self):
        """ + mha """
        self.layer_mha_input   =  {}
        self.layer_mha_wk      =  {}
        self.layer_mha_wq      =  {}
        self.layer_mha_wv      =  {}
        self.layer_mha_output  =  {}
        self.layer_mha_wo      =  {} 

    """ + reset layer_mlp_data"""
    def reset_layer_mlp_data(self):
        """ + mlp """
        self.layer_mlp_ifc1    =  {}
        self.layer_mlp_wfc1    =  {}
        self.layer_mlp_ifc2    =  {}
        self.layer_mlp_wfc2    =  {}

    """ + reset mlp_head_data"""
    def reset_mlp_head_data(self):
        """ + mlp head"""
        self.mlp_head_i        =  {}
        self.mlp_head_w        =  {}

    




    