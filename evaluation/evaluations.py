import torch
import wandb
import numpy as np
import gc
import tqdm
import random
from .evaluation_metrics import *
from train.training_handler import random_data_augmentation
from utils.sampling_utils import deform
from models.adapter_wrapper import AdapterWrapper

def evaluate_model(model,
                   dataloader,
                   params):
    '''
    Evaluate the model on the given dataloader.
    '''

    model.eval()
    correct = 0
    total = 0
    device = params.device
    model.to(device)
    dataiter = tqdm.tqdm(dataloader, total=len(dataloader))
    metric = get_metric(params)

    for data in dataiter:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        total += labels.size(0)
        if getattr(params, 'task', 'classification') == 'classification':
            correct += metric(outputs, labels) # unnormalized accuracy i.e., correct predictions count
        elif getattr(params, 'task', 'classification') == 'segmentation':
            correct += metric(outputs, labels) # mean_iou


    values_to_log = {
        'test_accuracy': correct / total
    }
    print(values_to_log)
    if params.wandb_log:
        wandb.log(values_to_log, commit=True)

    return correct / total


def consistency_test_segmentation(model,
                    dataset,
                    params):
    '''
    Calculate consistency evaluation metric for segmentation.
    It calculates the variance of the mean_iou metric for the same input image with different object sizes.
    Low variance means the model is consistent.
    '''
    model.eval()
    variance = 0
    total = 0
    device = params.device
    model.to(device)
    metric = get_metric(params)

    for i in range(len(dataset)):
        metric_list = []
        for s in range(params.test_augmentation_limit):
            inputs, labels = dataset.__getitem__(i, sub_idx=s)
            inputs, labels = inputs.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
            outputs = model(inputs)
            metric_list.append(metric(outputs, labels))
            del inputs, labels, outputs
            gc.collect()
            torch.cuda.empty_cache()
        variance += np.var(metric_list)
        total += 1

        if hasattr(params, 'test_samples') and i >= params.test_samples:
            break
    
    values_to_log = {"consistency_eq_error": variance / total}
    print(values_to_log)
    if params.wandb_log:
        wandb.log(values_to_log, commit=True)
        

def warping_equivariance_error(model,
                               dataloader,
                               params):
    '''
    Calculate the local scaling (scale by monotone function) equivariance error.
    Invarinace error for classification and equivariance error for segmentation.
    '''
    
    model.eval()
    device = params.device
    eq_error = 0
    total = 0
    model.to(device)

    dataiter = tqdm.tqdm(dataloader, total=len(dataloader))
    for data in dataiter:
        x, y = data[0].to(device), data[1].to(device)
        res = model(x)
        if params.task == 'classification':
            res = torch.softmax(res, dim=1)
        
        xp, yp, aug_x, aug_y = random_data_augmentation(x, y, params)
        res_p = model(xp)
        
        with torch.no_grad():
            if params.task == 'classification':
                res_p = torch.softmax(res_p, dim=1)
                error = torch.sum((res - res_p)**2, dim=-1) #/ torch.sum(res**2, dim=-1)
            elif params.task == 'segmentation':
                aug_res = deform(aug_x, aug_y, res)
                aug_res = torch.softmax(aug_res, dim=1)
                res_p = torch.softmax(res_p, dim=1)
                error = torch.sum((aug_res - res_p)**2, dim=[-1,-2,-3]) / (x.shape[-1] * x.shape[-2])
                del aug_res
            else:
                raise ValueError('Task not supported')
            
            eq_error += torch.sum(error**0.5).item()
            total += x.size(0)

        # del x, y, res, xp, yp, aug_x, aug_y, res_p
        # gc.collect()
        # torch.cuda.empty_cache()
    
    values_to_log = {
        'warp_eq_error': eq_error / total
    }
    print(values_to_log)
    if params.wandb_log:
        wandb.log(values_to_log, commit=True)
        

def local_scale_equivariance_testing(model,
                                     dataloader,
                                     params):
    
    '''
    Calculate the local scaling  equivariance error for Mnist multi digiot dataset.
    '''

    model.eval()
    device = params.device
    eq_error = 0
    total = 0
    model.to(device)

    for i in range(len(dataloader)):
        if i >= params.test_samples:
            break
        # Get the input data
        inputs, labels = dataloader.dataset.__getitem__(i, [1]*params.n_digits)
        # add batch dimension
        inputs = inputs.unsqueeze(0)

        res = model(inputs.to(device), return_features=True)
        res = torch.softmax(res, dim=1)


        for _ in np.arange( params.test_scale_count):
            
            scale_list = []
            for scale in params.test_s:
                rs = random.uniform(scale[0], scale[1])
                scale_list.append(rs)
            inputs, labels = dataloader.dataset.__getitem__(i, scale_list=scale_list)
            # add batch dimension
            inputs = inputs.unsqueeze(0)

            res_s = model(inputs.to(device), return_features=True)
            res_s = torch.softmax(res_s, dim=1)

            # Compare res and res_s
            error = torch.norm(res - res_s) / torch.norm(res)
            eq_error += error.item()
            total += 1

            del inputs, labels, res_s
            gc.collect()
            torch.cuda.empty_cache()
        
        if  hasattr(params, 'test_samples') and i >= params.test_samples:
            break

    values_to_log = {
        'lscal_eq_error': eq_error / total
    }
    print(values_to_log)
    if params.wandb_log:
        wandb.log(values_to_log, commit=True)

def evaluate_all_metric(model,
                        surrogate_model,
                        adapater_params,
                        test_dataloader,
                        test_dataset,
                        params):
    '''
    calculate all the evaluation metrics.
    model: the backbone model
    surrogate_model: the surrogate model (DEM/surrogate)
    adapater_params: the parameters of the adapter, object of Class PerlayerAdapterParams (in layers/adapter_params.py)
    test_dataloader: the dataloader for the test set
    test_dataset: the dataset for the test set 
    params: configuration
    '''

    wrapped_model = AdapterWrapper(
            model=model,
            surrogate_model=surrogate_model,
            phi_x_list=adapater_params.param_x_list if adapater_params is not None else None,
            phi_y_list=adapater_params.param_y_list if adapater_params is not None else None,
            params=params)
    
    evaluate_model(wrapped_model, test_dataloader, params)
    warping_equivariance_error(
        wrapped_model, test_dataloader, params)
    if params.data_set == 'mnist':
            local_scale_equivariance_testing(
                wrapped_model, test_dataloader, params)
    if params.task == 'segmentation':
        consistency_test_segmentation(
            wrapped_model, test_dataset, params)