"""
Project Name: Cifar100 Evaluation Base on VIT_B_16
Author: Zhangshize

Project Description:
This is a PyTorch implementation for training and evaluating on the Cifar100 dataset.
It includes implementations of quantized VIT_B_16.

8-bit SLFP and 7-bit SFP quantization based on max-scaling are implemented.

Dependencies:
- Python 3.6+
- PyTorch 1.0+
- torchvision 0.2.2+

Installation and Running:
1. Clone this repository
2. Run the code: python ./eval.py --Qbits <bit width>

"""
import torch
import argparse
import torchvision
""" + ViT_B16_QAT"""
from models.ViT_B16_QAT import VisionTransformer
""" + get_scale_factor"""
from utils.get_scale_factor_utils import *
from torch.utils import data
from torchvision import transforms
from config.ViT_config import ViT_Config
from utils.evaluation_utils import evaluation

parser = argparse.ArgumentParser(description='SLFP reference and retrain, pytorch implementation')
parser.add_argument('--pre_reference', action='store_true', default=False)  
parser.add_argument('--Qbits', type=int, default=32)      

cfg = parser.parse_args()  

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def main():
    vit_config = ViT_Config()
    test_model = VisionTransformer(vit_config, load_head=False, qbits = cfg.Qbits, pre_reference = cfg.pre_reference)

    
    """ + loading weight model .pth"""
    MODEL_PATH = './ckpt/cifar100/MLP-PEFT-acc0.9248-Epoch20.pth'                             # fp32
    MODEL_PATH = './ckpt/cifar100/ViT-b16-CIFAR100-Epoch10-PEFT-SLFP-QAT-acc0.9168.pth'       # slfp

    test_model.load_state_dict(torch.load(MODEL_PATH))

    """ + loading weight model  .npz"""
    MODEL_PATH = './ckpt/cifar100/ViT-B_16.npz'
    test_model.load_weights(np.load(MODEL_PATH))


    """ ------ test ------- """
    """ cifar100 """
    batch_size = 64
    DATA_PATH = './data/cifar100'
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224,scale=(0.64,1.0),ratio=(1.0,1.0)),
        # transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    test_data = torchvision.datasets.CIFAR100(
        root=DATA_PATH,
        train=False,
        transform=trans,
        download=True)
    test_iter = data.DataLoader(test_data, batch_size, shuffle=True)


    """ + pre_reference"""
    if cfg.pre_reference:
        folder_path = "./scale_factor/"

        max_abs_layer_mha_input, max_abs_layer_mha_wk, max_abs_layer_mha_wq, max_abs_layer_mha_wv, max_abs_layer_mha_output, max_abs_layer_mha_wo, max_abs_layer_mlp_ifc1, max_abs_layer_mlp_wfc1, max_abs_layer_mlp_ifc2, max_abs_layer_mlp_wfc2, max_abs_mlp_head_i, max_abs_mlp_head_w = get_layer_scale_factor(test_model, test_iter, 1000, try_gpu())

        put_mha_scale_factor_layer_txt(max_abs_layer_mha_input, max_abs_layer_mha_wk, max_abs_layer_mha_wq, max_abs_layer_mha_wv, max_abs_layer_mha_output, max_abs_layer_mha_wo, folder_path)
        put_mlp_scale_factor_layer_txt(max_abs_layer_mlp_ifc1, max_abs_layer_mlp_wfc1, max_abs_layer_mlp_ifc2, max_abs_layer_mlp_wfc2, folder_path)
        put_mlp_head_scale_factor_txt(max_abs_mlp_head_i, max_abs_mlp_head_w, folder_path)


    test_acc = evaluation(test_model, test_iter, try_gpu())
    print(f"Test Accuracy: {test_acc:.2%}")

if __name__ == '__main__':
    main()



   

