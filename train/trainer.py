import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from time import time as default_timer
from layers.adapter import *
import tqdm
from models.adapter_wrapper import AdapterWrapper
import os
from evaluation.evaluations import *
import wandb
import random
from evaluation.evaluation_metrics import *
from .training_handler import *



def bi_level_train(*,
                    model,
                    surrogate_model,
                    phi_x_list,
                    phi_y_list,
                    train_loader,
                    test_loader,
                    test_dataset,
                    params,
                    trouble_shoot=False
                    ):
    """
    Train a model with bi-level optimization. Depending of the configuration, it optimized the following objectives:
      phi_x*, phi_y* = arg_min_{phi_x, phi_y} f(x, phi_x, phi_y)
      \theta = argmin_{\theta} L(Model_(theta)(Local_scale_(x, phi_x*, phi_y*), y)

    where 
        f is the surrogate model/DEM model
        model is the backbone model
        L is the loss function (e.g., cross entropy loss)
        Local_scale_(x, phi_x*, phi_y*) is the local scale of the input x, which is obtained by applying the canonicalization parameters phi_x* and phi_y* to the input x
        Model_(theta) is the model with parameters theta
        y is the target output

    The model is trained with bi-level optimization. The outer loop is the outer optimization loop, and the inner loop is the inner optimization loop.

    
    args:
        model : backbone model
        surrogate_model : model to be used for DEM/inner optimization
        
        phi_x_list : list of augmentation parameters for each layers of model (along x axis)
                    each element should be of shape (1, R, R-1)
        phi_y_list : list of augmentation parameters for each layers of model (along y axis)

        train_loader : dataloader for training data
        test_loader : dataloader for testing data
        test_dataset : dataset for testing data
        params : parameters object
        trouble_shoot : if True, return after first iteration
    """

    log_interval = params.wandb_log_interval
    wandb_log = params.wandb_log
    epochs = params.epochs
    # get optimizers
    outer_optim = get_optimizer(params, model, params.outer_lr, params.outer_optimizer_kwargs)
    outer_scheduler = get_scheduler(params, outer_optim)
    
    # get surrogate optimizer and scheduler
    if params.surrogate_model:
        surrogate_optim = get_optimizer(params, surrogate_model, params.surrogate_lr, params.surrogate_optimizer_kwargs)
        surrogate_scheduler = get_scheduler(params, surrogate_optim)
    else:
        surrogate_optim = None
        surrogate_scheduler = None
    
    loss_function = get_loss(params)  # cross entropy loss
    metric = get_metric(params)  # accuracy or mIoU


    device = params.device
    model = model.to(device)
    if surrogate_model is not None: surrogate_model = surrogate_model.to(device)
    optmizer_handler = OptimizerHandler(params)

    for ep in range(epochs):
        optmizer_handler.train(model, surrogate_model, ep)
        t1 = default_timer()
        train_loss, train_steps, train_acc, num_samples = 0, 0, 0, 0
        equi_loss, unique_loss = 0, 0

        train_loader_iter = tqdm.tqdm(train_loader, desc=f'Epoch {ep}/{epochs}', leave=False, ncols=100)

        for data in train_loader_iter:
            images, labels = data
            Batch_Size = images.size(0)
            # data augmentation
            data_aug_prob = random.random()
            # data augmentation is done with 50% probability
            if (getattr(params, "do_data_augmentation", False) and data_aug_prob < 0.5) \
                or getattr(params, "unique_optima_loss", False):
                with torch.no_grad():
                    images = images.to(device)
                    labels = labels.to(device)
                    images_aug, label_aug, phi_x_aug, phi_y_aug = random_data_augmentation(images.clone().detach(), labels.clone().detach(), params)                
                    if getattr(params, "unique_optima_loss", False):
                        # for unique optima loss we need to use augmented data so it is concatenated with the original data
                        images, labels = torch.cat([images, images_aug], dim=0), torch.cat([labels, label_aug], dim=0)
                    else:
                        images, labels = images_aug, label_aug

            else:
                images, labels = images.to(device), labels.to(device)

            phi_x_batch, phi_y_batch = None, None
            
            if phi_x_list is not None and params.adaptation_pause<=ep: # not doing adaption in the first adaptation_pause epochs
                optmizer_handler.zero_grad(outer_optim, surrogate_optim, ep)

                # inner optimization
                phi_x_batch, phi_y_batch, out_list = inner_optimization(model=model,
                                                              canonicalizer_model=surrogate_model,
                                                              images=images,
                                                              params=params,
                                                              phi_x_list=phi_x_list,
                                                              phi_y_list=phi_y_list,
                                                              create_grad_graph=True,
                                                              wandb_log=params.wandb_log)

                # forward pass
                yp = model(
                    images,
                    phi_x_list=phi_x_batch,
                    phi_y_list=phi_y_batch,
                    skip_input_augmentation=getattr(params, "skip_input_augmentation", False))
                
                if getattr(params, "unique_optima_loss", False) and getattr(params, "do_data_augmentation", False):
                    loss = loss_function(yp[Batch_Size:], labels[Batch_Size:]) if data_aug_prob < 0.5 \
                        else loss_function(yp[:Batch_Size], labels[:Batch_Size])
                else:
                    loss = loss_function(yp, labels)
                


                loss = weight_task_loss(loss, params, optmizer_handler.current_model_under_training)
                train_loss += loss.item()
                
                if getattr(params, "equivariance_loss", False):
                    if getattr(params, "unique_optima_loss", False):
                        error = calculate_eq_loss(yp[:Batch_Size], yp[Batch_Size:], phi_x_aug, phi_y_aug, params)
                    else:
                        error = get_equivariance_loss(images, labels, model, surrogate_model, params, yp, phi_x_list, phi_y_list, inner_optimization)
                    mean_eq_error = torch.mean(error)
                    equi_loss += mean_eq_error.item()
                    loss += weight_equivariance_loss(mean_eq_error, params, optmizer_handler.current_model_under_training)
                
                # the following loss is used to encourage the model to find unique optima
                if getattr(params, "unique_optima_loss", False):
                    mean_unique_loss = get_unique_optima_loss(out_list, phi_x_batch, phi_y_batch,Batch_Size, params, ep)
                    unique_loss +=mean_unique_loss.item()
                    loss += mean_unique_loss

                loss.backward()
                if trouble_shoot:
                    return
            else:
                optmizer_handler.zero_grad(outer_optim, surrogate_optim, ep)
                yp = model(images)
                loss = loss_function(yp, labels)
                train_loss += loss.item()
                if getattr(params, "equivariance_loss", False):
                    error = get_equivariance_loss(images, labels, model, surrogate_model, params, yp, phi_x_list, phi_y_list, inner_optimization)
                    mean_eq_error = torch.mean(error)
                    equi_loss += mean_eq_error.item()
                    loss += getattr(params, "equivariance_loss_weight", 0.1) * mean_eq_error
                loss.backward()

            # get train accuracy
            with torch.no_grad():
                # _, predicted = torch.max(yp.data, 1)
                train_acc += metric(yp.data[:Batch_Size], labels[:Batch_Size])#(predicted == y).sum().item()
                num_samples += labels[:Batch_Size].size(0)

            if params.clip_gradient:
                optmizer_handler.clip_grad(model, surrogate_model)
            optmizer_handler.step(outer_optim, surrogate_optim, ep)
            train_steps += 1

        # torch.cuda.empty_cache()
        avg_train_l2 = train_loss / train_steps
        equi_loss = equi_loss / train_steps
        unique_loss = unique_loss / train_steps

        optmizer_handler.scheduler_step(outer_scheduler, surrogate_scheduler, ep, avg_train_l2)
        t2 = default_timer()
        epoch_train_time = t2 - t1

        if ep % log_interval == 0:
            # loging to wandb
            values_to_log = {
                'train_err': avg_train_l2, 'time': epoch_train_time, 'train_acc': train_acc / num_samples, 
                'c_lr': outer_optim.param_groups[0]['lr']
            }
            if getattr(params, "equivariance_loss", False) or getattr(params, 'unique_optima_loss', False):
                values_to_log.update({'equi_loss': equi_loss, 'unique_loss': unique_loss})
            if surrogate_optim is not None:
                values_to_log.update({'s_lr': surrogate_optim.param_groups[0]['lr']})
            print(f"Epoch {ep}: Time: {epoch_train_time:.2f}s, Loss: {avg_train_l2:.4f}, Train Accuracy: {train_acc / num_samples:.4f}")
            if wandb_log:
                wandb.log(values_to_log, commit=True)

        if ep % params.weight_saving_interval == 0 or ep == epochs - 1:
            torch.save(model.state_dict(), os.path.join(params.save_model_path, f"{ep} " + params.config + "_" + params.task + ".pth"))
            torch.save(phi_x_list, os.path.join(params.save_model_path, f"{ep} " + params.config + "_" + params.task + "_phi_x.pth"))
            torch.save(phi_y_list, os.path.join(params.save_model_path, f"{ep} " + params.config + "_" + params.task + "_phi_y.pth"))
            if params.surrogate_model: torch.save(surrogate_model.state_dict(), os.path.join(params.save_model_path, f"{ep} " + params.config + "_" + params.task + "_surrogate.pth"))

            # doing all the evaluations
            wrapped_model = AdapterWrapper(model=model,
                                          surrogate_model=surrogate_model, 
                                          phi_x_list=phi_x_list if (phi_x_list is not None and params.adaptation_pause <= ep) else None,
                                          phi_y_list=phi_y_list if (phi_y_list is not None and params.adaptation_pause <= ep) else None,
                                          params=params)

            evaluate_model(wrapped_model, test_loader, params)

            if ep % (2*params.weight_saving_interval) == 0 or ep == epochs - 1:
                if params.data_set == 'mnist':
                    local_scale_equivariance_testing(
                        wrapped_model, test_loader, params)
                if params.task == 'segmentation':
                    consistency_test_segmentation(
                        wrapped_model, test_dataset, params)


