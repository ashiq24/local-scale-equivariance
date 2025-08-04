import torch
from train.training_handler import *
from collections import defaultdict
import numpy as np
from timm import utils

def get_unique_loss(out_x, out_y, out_x_aug, out_y_aug, epoch, params):
    
    # merge out_x and out_x_aug along the batch dimension
    out_x_com = [torch.cat([out_x[i], out_x_aug[i]]) for i in range(len(out_x))]
    out_y_com = [torch.cat([out_y[i], out_y_aug[i]]) for i in range(len(out_y))]
    Batch_size = out_x[0].shape[0]
    
    return get_unique_optima_loss(None, out_x_com, out_y_com, Batch_size, params, epoch=epoch+1, return_sep_loss=True)



def group_image_indices(data):
    groups = defaultdict(list)
    
    # Step 1: Group indices by (class_dir, base_image)
    for idx, (filepath, _) in enumerate(data):
        parts = filepath.split('/')
        class_dir = parts[-2]  # e.g., 'n01440764'
        filename = parts[-1]   # e.g., 'ILSVRC2012_val_00025527_scale_0.7.JPEG'
        base_image = filename.split('_scale_')[0]  # e.g., 'ILSVRC2012_val_00025527'
        key = (class_dir, base_image)
        groups[key].append(idx)
    
    # Step 2: Reorder indices within each group to prioritize scale=1.0 at index 0
    ordered_groups = []
    for group in groups.values():
        scale_indices = []
        # Extract scale values for sorting
        for idx in group:
            filepath = data[idx][0]  # Get the filepath from the original data
            filename = filepath.split('/')[-1]
            scale_part = filename.split('_scale_')[1][:-5]  # discard '.JPEG'
            scale = float(scale_part) if scale_part != '1' else 1.0  # Convert to float
            scale_indices.append((scale, idx))
        
        # Split into scale=1.0 and others
        scale_1 = [item for item in scale_indices if item[0] == 1.0]
        others = [item for item in scale_indices if item[0] != 1.0]
        
        # Sort others by ascending scale (0.7 < 0.8 < 0.9 < 1.1 < 1.2 < 1.3)
        others_sorted = sorted(others, key=lambda x: x[0])
        
        # Combine: scale=1.0 first, then sorted others
        ordered_group = scale_1 + others_sorted
        
        # Extract the indices in the new order
        ordered_indices = [idx for (scale, idx) in ordered_group]
        ordered_groups.append(ordered_indices)
    
    return ordered_groups

def get_image_batch(data_loader, indices):
    images = []
    tergets = []
    for idx in indices:
        image, terget = data_loader.dataset[idx]
        images.append(torch.tensor(image))
        tergets.append(torch.tensor(terget))
        
    return torch.stack(images), torch.stack(tergets)

def filter_groups_and_reshape(grouped_indices, batch_size):
    lens = [len(x) for x in grouped_indices]
    max_len = max(set(lens), key = lens.count)
    grouped_indices_np = [x for x in grouped_indices if len(x) == max_len]
    merge_rows = batch_size // max_len
    
    actual_batch_size = max_len * merge_rows
    n_batches = len(grouped_indices_np) // actual_batch_size
    grouped_indices_np = grouped_indices_np[:n_batches * actual_batch_size]
    
    grouped_indices_np = np.array(grouped_indices_np)
    merge_rows = batch_size // max_len
    grouped_indices_np = grouped_indices_np.reshape(-1, merge_rows * max_len)
    
    return grouped_indices_np, max_len, merge_rows

def calculate_eq_loss_eval(outputs, merge_rows, max_len):
    outputs = torch.softmax(outputs, dim=1)
    outputs = outputs.reshape(merge_rows, max_len, 1000)
    # get first element of each row
    output_indentity = outputs[:, 0, ...]
    # subtract each row by the first element of the row
    outputs = outputs - output_indentity.unsqueeze(1)
    # norm each row
    outputs = torch.norm(outputs, p=2, dim=2)
    # print("Norm values ", outputs.shape)
    return torch.mean(outputs)

def per_scale_acc(output, target, merge_rows, max_len):
    output = output.clone()
    target = target.clone()
    output = output.reshape(merge_rows, max_len, 1000)
    target = target.reshape(merge_rows, max_len)
    per_scale_acc =[]
    for i in range(max_len):
        scale_output = output[:, i, ...]
        scale_target = target[:, i]
        acc = utils.accuracy(scale_output, scale_target, (1,5))
        per_scale_acc.append([i.item() for i in acc])
    per_scale_acc = torch.tensor(per_scale_acc)
    return per_scale_acc