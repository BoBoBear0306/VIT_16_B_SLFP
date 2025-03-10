import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


""" + layer scale factor of transformer """
def get_layer_scale_factor(model, data_loader, total_images, device): # get Ka, Kw
    if isinstance(model, nn.Module):
        model.eval()
        model.to(device)
    print(f"evaluation on: {device}")

    """ + layer_mha_data"""
    layer_mha_input = {}
    layer_mha_wk    = {}
    layer_mha_wq    = {}
    layer_mha_wv    = {}
    layer_mha_output= {}
    layer_mha_wo    = {}

    """ + layer_mlp_data"""
    layer_mlp_ifc1  = {}
    layer_mlp_wfc1  = {}
    layer_mlp_ifc2  = {}
    layer_mlp_wfc2  = {}

    """ + mlp_head_data"""
    mlp_head_i  = []
    mlp_head_w  = []




    
    # get i, k, q, v, o and wo for total_images' mha , get ifc1, wfc1, ifc2, wfc2 for total_images' mlp, get input and w for  mlp_head_data
    i = 0
    count = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        i = i + 1
        
        inputs, targets = inputs.cuda(), targets.cuda()

        # record i, k, q, v, o and wo for each layer, mha
        model.reset_layer_mha_data()
        # record ifc1, wfc1, ifc2, wfc2 for each layer, mlp
        model.reset_layer_mlp_data()
        # record input and weight for mlp head
        model.reset_mlp_head_data()


        # test forward
        outputs = model(inputs)

        # mha, get i, k, q, v, o and wo for each layer
        current_layer_mha_input, current_layer_mha_wk, current_layer_mha_wq, current_layer_mha_wv, current_layer_mha_output, current_layer_mha_wo = model.get_layer_mha_data()
        # mlp, get ifc1, wfc1, ifc2, wfc2 for total_images' mlp
        current_layer_mlp_ifc1, current_layer_mlp_wfc1, current_layer_mlp_ifc2, current_layer_mlp_wfc2 = model.get_layer_mlp_data()
        # mlp head, get input and weight 
        current_mlp_head_i, current_mlp_head_w = model.get_mlp_head_data()


        # mha 
        for idx, tensor in current_layer_mha_input.items():
            if idx not in layer_mha_input:
                layer_mha_input[idx] = []
            layer_mha_input[idx].append(tensor.detach().cpu())

        for idx, tensor in current_layer_mha_wk.items():
            if idx not in layer_mha_wk:
                layer_mha_wk[idx] = []
            layer_mha_wk[idx].append(tensor.detach().cpu())     

        for idx, tensor in current_layer_mha_wq.items():
            if idx not in layer_mha_wq:
                layer_mha_wq[idx] = []
            layer_mha_wq[idx].append(tensor.detach().cpu())   

        for idx, tensor in current_layer_mha_wv.items():
            if idx not in layer_mha_wv:
                layer_mha_wv[idx] = []
            layer_mha_wv[idx].append(tensor.detach().cpu())  

        for idx, tensor in current_layer_mha_output.items():
            if idx not in layer_mha_output:
                layer_mha_output[idx] = []
            layer_mha_output[idx].append(tensor.detach().cpu())  

        for idx, tensor in current_layer_mha_wo.items():
            if idx not in layer_mha_wo:
                layer_mha_wo[idx] = []
            layer_mha_wo[idx].append(tensor.detach().cpu())  
        
        # mlp
        for idx, tensor in current_layer_mlp_ifc1.items():
            if idx not in layer_mlp_ifc1:
                layer_mlp_ifc1[idx] = []
            layer_mlp_ifc1[idx].append(tensor.detach().cpu())   

        for idx, tensor in current_layer_mlp_wfc1.items():
            if idx not in layer_mlp_wfc1:
                layer_mlp_wfc1[idx] = []
            layer_mlp_wfc1[idx].append(tensor.detach().cpu())  

        for idx, tensor in current_layer_mlp_ifc2.items():
            if idx not in layer_mlp_ifc2:
                layer_mlp_ifc2[idx] = []
            layer_mlp_ifc2[idx].append(tensor.detach().cpu())  

        for idx, tensor in current_layer_mlp_wfc2.items():
            if idx not in layer_mlp_wfc2:
                layer_mlp_wfc2[idx] = []
            layer_mlp_wfc2[idx].append(tensor.detach().cpu()) 

        # mlp head
        mlp_head_i.append(current_mlp_head_i.detach().cpu()) 
        mlp_head_w.append(current_mlp_head_w.detach().cpu()) 

        count += len(inputs)
        if count >= total_images:
            break 

    # calculate the maximum absolute values for layer i, k, q, v, o and wo    
    # mha
    max_abs_layer_mha_input  = []
    max_abs_layer_mha_wk     = []
    max_abs_layer_mha_wq     = []
    max_abs_layer_mha_wv     = []
    max_abs_layer_mha_output = []
    max_abs_layer_mha_wo     = []
    # mlp
    max_abs_layer_mlp_ifc1   = []
    max_abs_layer_mlp_wfc1   = []
    max_abs_layer_mlp_ifc2   = []
    max_abs_layer_mlp_wfc2   = []
    # mlp head
    max_abs_mlp_head_i       = []
    max_abs_mlp_head_w       = []


    # mha
    for idx, list in layer_mha_input.items():
        max_abs_data = torch.max(torch.abs(torch.cat(list, dim=0))) 
        max_abs_layer_mha_input.append(max_abs_data.item()) 

    for idx, list in layer_mha_wk.items():
        max_abs_data = torch.max(torch.abs(torch.cat(list, dim=0))) 
        max_abs_layer_mha_wk.append(max_abs_data.item()) 

    for idx, list in layer_mha_wq.items():
        max_abs_data = torch.max(torch.abs(torch.cat(list, dim=0))) 
        max_abs_layer_mha_wq.append(max_abs_data.item()) 

    for idx, list in layer_mha_wv.items():
        max_abs_data = torch.max(torch.abs(torch.cat(list, dim=0))) 
        max_abs_layer_mha_wv.append(max_abs_data.item()) 

    for idx, list in layer_mha_output.items():
        max_abs_data = torch.max(torch.abs(torch.cat(list, dim=0))) 
        max_abs_layer_mha_output.append(max_abs_data.item()) 

    for idx, list in layer_mha_wo.items():
        max_abs_data = torch.max(torch.abs(torch.cat(list, dim=0))) 
        max_abs_layer_mha_wo.append(max_abs_data.item()) 
    
    # mlp
    for idx, list in layer_mlp_ifc1.items():
        max_abs_data = torch.max(torch.abs(torch.cat(list, dim=0))) 
        max_abs_layer_mlp_ifc1.append(max_abs_data.item()) 

    for idx, list in layer_mlp_wfc1.items():
        max_abs_data = torch.max(torch.abs(torch.cat(list, dim=0))) 
        max_abs_layer_mlp_wfc1.append(max_abs_data.item()) 

    for idx, list in layer_mlp_ifc2.items():
        max_abs_data = torch.max(torch.abs(torch.cat(list, dim=0))) 
        max_abs_layer_mlp_ifc2.append(max_abs_data.item()) 

    for idx, list in layer_mlp_wfc2.items():
        max_abs_data = torch.max(torch.abs(torch.cat(list, dim=0))) 
        max_abs_layer_mlp_wfc2.append(max_abs_data.item()) 

    # mlp head
    max_abs_mlp_head_i = torch.max(torch.abs(torch.cat(mlp_head_i, dim=0))) 
    max_abs_mlp_head_w = torch.max(torch.abs(torch.cat(mlp_head_w, dim=0))) 


    

    return  max_abs_layer_mha_input, max_abs_layer_mha_wk, max_abs_layer_mha_wq, max_abs_layer_mha_wv, max_abs_layer_mha_output, max_abs_layer_mha_wo, max_abs_layer_mlp_ifc1, max_abs_layer_mlp_wfc1, max_abs_layer_mlp_ifc2, max_abs_layer_mlp_wfc2, max_abs_mlp_head_i, max_abs_mlp_head_w




""" + mha, layer scale factor of transformer to txt"""
def put_mha_scale_factor_layer_txt(max_abs_layer_mha_input, max_abs_layer_mha_wk, max_abs_layer_mha_wq, max_abs_layer_mha_wv, max_abs_layer_mha_output, max_abs_layer_mha_wo, folder_path):
    
    """ ---------------  i, k, q, v, o and wo ----------------"""
    # i
    with open(folder_path + 'max_layer_mha_input.txt', 'w') as file:
        for item in max_abs_layer_mha_input:
            file.write(str(item) + '\n')

    # k
    with open(folder_path + 'max_layer_mha_wk.txt', 'w') as file:
        for item in max_abs_layer_mha_wk:
            file.write(str(item) + '\n')

    # q
    with open(folder_path + 'max_layer_mha_wq.txt', 'w') as file:
        for item in max_abs_layer_mha_wq:
            file.write(str(item) + '\n')   

    # v
    with open(folder_path + 'max_layer_mha_wv.txt', 'w') as file:
        for item in max_abs_layer_mha_wv:
            file.write(str(item) + '\n') 

    # o
    with open(folder_path + 'max_layer_mha_output.txt', 'w') as file:
        for item in max_abs_layer_mha_output:
            file.write(str(item) + '\n')        

    # wo
    with open(folder_path + 'max_layer_mha_wo.txt', 'w') as file:
        for item in max_abs_layer_mha_wo:
            file.write(str(item) + '\n')              

""" + mlp, layer scale factor of transformer to txt"""
def put_mlp_scale_factor_layer_txt(max_abs_layer_mlp_ifc1, max_abs_layer_mlp_wfc1, max_abs_layer_mlp_ifc2, max_abs_layer_mlp_wfc2, folder_path):

    """ ---------------  ifc1, wfc1, ifc2, wfc2 ----------------"""
    # ifc1
    with open(folder_path + 'max_layer_mlp_ifc1.txt', 'w') as file:
        for item in max_abs_layer_mlp_ifc1:
            file.write(str(item) + '\n')
    # wfc1
    with open(folder_path + 'max_layer_mlp_wfc1.txt', 'w') as file:
        for item in max_abs_layer_mlp_wfc1:
            file.write(str(item) + '\n')
    # ifc2
    with open(folder_path + 'max_layer_mlp_ifc2.txt', 'w') as file:
        for item in max_abs_layer_mlp_ifc2:
            file.write(str(item) + '\n')
    # wfc2
    with open(folder_path + 'max_layer_mlp_wfc2.txt', 'w') as file:
        for item in max_abs_layer_mlp_wfc2:
            file.write(str(item) + '\n')

""" + mlp head, scale factor of transformer to txt"""
def put_mlp_head_scale_factor_txt(max_abs_mlp_head_i, max_abs_mlp_head_w, folder_path):

    """ ---------------  input, weight ----------------"""
    # input 
    with open(folder_path + 'max_mlp_head_input.txt', 'w') as file:
        file.write(str(max_abs_mlp_head_i.item()) + '\n')

    # weight 
    with open(folder_path + 'max_mlp_head_weight.txt', 'w') as file:
        file.write(str(max_abs_mlp_head_w.item()) + '\n')


""" + mha, get layer scale factor of transformer for txt"""
def acquire_mha_layer_scale_factor_txt(folder_path):
    # Read the txt file and convert the data to a list form

    # i
    with open(folder_path + 'max_layer_mha_input.txt', 'r') as file:
        max_abs_layer_mha_input = [float(line.strip()) for line in file]  

    # k
    with open(folder_path + 'max_layer_mha_wk.txt', 'r') as file:
        max_abs_layer_mha_wk = [float(line.strip()) for line in file] 

    # q
    with open(folder_path + 'max_layer_mha_wq.txt', 'r') as file:
        max_abs_layer_mha_wq = [float(line.strip()) for line in file] 

    # v
    with open(folder_path + 'max_layer_mha_wv.txt', 'r') as file:
        max_abs_layer_mha_wv = [float(line.strip()) for line in file] 

    # o
    with open(folder_path + 'max_layer_mha_output.txt', 'r') as file:
        max_abs_layer_mha_output = [float(line.strip()) for line in file] 

    # wo
    with open(folder_path + 'max_layer_mha_wo.txt', 'r') as file:
        max_abs_layer_mha_wo = [float(line.strip()) for line in file] 

    return  max_abs_layer_mha_input, max_abs_layer_mha_wk, max_abs_layer_mha_wq, max_abs_layer_mha_wv, max_abs_layer_mha_output, max_abs_layer_mha_wo


""" + mlp, get layer scale factor of transformer for txt"""
def acquire_mlp_layer_scale_factor_txt(folder_path):
    # Read the txt file and convert the data to a list form

    # ifc1
    with open(folder_path + 'max_layer_mlp_ifc1.txt', 'r') as file:
        max_abs_layer_mlp_ifc1 = [float(line.strip()) for line in file] 

    # wfc1
    with open(folder_path + 'max_layer_mlp_wfc1.txt', 'r') as file:
        max_abs_layer_mlp_wfc1 = [float(line.strip()) for line in file] 
    
    # ifc2
    with open(folder_path + 'max_layer_mlp_ifc2.txt', 'r') as file:
        max_abs_layer_mlp_ifc2 = [float(line.strip()) for line in file]

    # ifc2
    with open(folder_path + 'max_layer_mlp_wfc2.txt', 'r') as file:
        max_abs_layer_mlp_wfc2 = [float(line.strip()) for line in file] 

    return  max_abs_layer_mlp_ifc1, max_abs_layer_mlp_wfc1, max_abs_layer_mlp_ifc2, max_abs_layer_mlp_wfc2
