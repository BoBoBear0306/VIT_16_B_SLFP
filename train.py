"""
Project Name: Cifar100 Training Base on VIT_B_16
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
2. Run the code: python ./train.py --Qbits <bit width>

"""

import torch
import argparse
import torchvision
import numpy as np
# from models.ViT_B16 import VisionTransformer
""" + ViT_B16_QAT"""
from models.ViT_B16_QAT import VisionTransformer
from torch.utils import data
from torchvision import transforms
from config.ViT_config import ViT_Config
from utils.train_utils import train_model, save_model, try_gpu

""" + argument"""
parser = argparse.ArgumentParser(description='SLFP reference and retrain, pytorch implementation')
parser.add_argument('--pre_reference', action='store_true', default=False)  
parser.add_argument('--Qbits', type=int, default=32)      

cfg = parser.parse_args()  

def data_loader(batch_size, train_trans, test_trans, data_path, shuffle=True):
    train_dataset = torchvision.datasets.CIFAR100(
        data_path,
        True,
        train_trans,
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR100(
        data_path,
        False,
        test_trans,
        download=True
    )
    train_iter = data.DataLoader(train_dataset, batch_size, shuffle=shuffle)
    test_iter = data.DataLoader(test_dataset, batch_size, shuffle=shuffle)
    return train_iter, test_iter

""" + imgaenet1k transform"""
def imgnet_transform(is_training=True):
  if is_training:
    transform_list = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ColorJitter(brightness=0.5,
                                                                contrast=0.5,
                                                                saturation=0.3),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
  else:
    transform_list = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
  return transform_list


def main():
    
    """ cifar100 """
    DATA_PATH = "./data"
    trans1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224,scale=(0.64,1.0),ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    trans2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224,scale=(0.64,1.0),ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    batch_size = 32
    train_iter, test_iter = data_loader(batch_size,
                                        trans1,
                                        trans2,
                                        DATA_PATH)


    # load pre-trained weights
    vit_config = ViT_Config()
    # change head for classification on CIFAR-100

    """ + qbits and pre_reference"""
    model = VisionTransformer(vit_config, load_head=False, qbits = cfg.Qbits, pre_reference = cfg.pre_reference)

    """ - pretrain fp32 from imagnet21k"""
    # model.load_weights(np.load('./ckpt/cifar100/ViT-B_16.npz'))
    
    """ + pretrain from PEFT"""
    model.load_state_dict(torch.load("./ckpt/cifar100/MLP-PEFT-acc0.9248-Epoch20.pth"))

    
    
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # unfreeze head
    model.mlp_head.weight.requires_grad = True
    model.mlp_head.bias.requires_grad = True

    # unfreeze mlp layers in encoder
    for i, layer in enumerate(model.feature_layer.encoder_layer):
        layer.mlp_norm.weight.requires_grad = True
        layer.mlp_norm.bias.requires_grad = True
        layer.mlp.fc1.weight.requires_grad = True
        layer.mlp.fc1.bias.requires_grad = True
        layer.mlp.fc2.weight.requires_grad = True
        layer.mlp.fc2.bias.requires_grad = True
    


    # save model weights
    datasets_name = 'cifar100'
    model_weights = 'ViT-b16-CIFAR100-Epoch10-PEFT-SLFP-QAT-acc'            # qta slfp
    # model_weights = 'MLP-PEFT-acc'                                        # fp32
    PATH = f'./ckpt/{datasets_name}/{model_weights}.pth'

    num_epochs = 20

    train_model(
            net=model,
            train_iter=train_iter,
            test_iter=test_iter,
            num_epochs=num_epochs,
            lr=1e-3,
            device=try_gpu(),
            PATH = PATH,
            test=True,
            plot=True,
            )
    

if __name__ == '__main__':
    main()
    
