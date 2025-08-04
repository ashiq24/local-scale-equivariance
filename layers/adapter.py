import torch
import torch.nn.functional as F
from utils.sampling_utils import *
from utils.stft import *
from torch import nn
from train.training_handler import *
import wandb
import random
from .optim import CustomAdam
from torch.optim import Adam


def inner_optimization(*,
                       model,
                       canonicalizer_model,
                       images,
                       params,
                       phi_x_list,
                       phi_y_list,
                       create_grad_graph,
                       wandb_log):
    '''
    model: backbone model
    canonicalizer_model: model to be used for canonicalization. This model will be used in the inner optimization loop, either in 
    the DEQ mode or the inner optimization loop.  
    images: input images
    params: params object
    phi_x_list: list of phi_x parameters for local scaling
    phi_y_list: list of phi_y parameters for local scaling
    create_grad_graph: create grad graph, only true during training
    wandb_log: log to wandb

    returns updated phi_x_batch and phi_y_batch, local scaling parameters for each images in the batch.
    '''

    # chcek if the backbone model is in training mode
    is_model_training = model.training

    if is_model_training:
        # this is to temporarily turn off dropout and batchnorm statistics calculation
        # for inner optimization loop through the canonicalizer model.
        model.eval()

    mode = '' # for logging
    output_list = []
    inner_epochs = params.inner_epochs
    if not create_grad_graph:
        mode = 'test_'
        inner_epochs = params.test_inner_epochs
        
    # expand the param_x and param_y to the batch size
    batch_size = images.size(0)
    phi_x_batch = [phi_x.clone().repeat(batch_size, 1, 1).requires_grad_(True).to(params.device) for phi_x in phi_x_list]
    phi_y_batch = [phi_y.clone().repeat(batch_size, 1, 1).requires_grad_(True).to(params.device) for phi_y in phi_y_list]

    # get the inner loss function
    inner_loss = get_inner_loss(params)

    if params.deep_equlibrium_mode:
        # For Deep Equilibrium Model, we need to run the DEQ interative forward pass to get the output.
        phi_x_batch, phi_y_batch = canonicalizer_model._DEQ(images, phi_x_batch, phi_y_batch)
    else:
        # If not DEQ mode, we do few steps of inner optimization.
        if params.inner_optimizer == 'adam':
            optim = CustomAdam(phi_x_batch + phi_y_batch, lr=params.inner_lr, betas= getattr(params, 'adam_betas', (0.9, 0.999)))
        
        intial_loss = final_loss = 0

        for innner_ep in range(1, inner_epochs + 1, 1):
            if params.surrogate_model:
                # optimization is done through the canonicalizer model.
                # phi_x_batch and phi_y_batch = argmin_phi_x, phi_y Canon_model(x, phi_x, phi_y)
                yp = canonicalizer_model(images, phi_x_batch=phi_x_batch, phi_y_batch=phi_y_batch)
            else:
                # here canonicalizer = main model, i.e., we directly optimize the output 
                # of the main model. Such as minimizing the entropy of the prediction.
                yp = model(
                    images,
                    phi_x_list=phi_x_batch,
                    phi_y_list=phi_y_batch,
                    skip_input_augmentation=getattr(params, "skip_input_augmentation", False))

            if params.surrogate_model:
                # surrogate model directly returns the scaler to optimize. So no need to use  the loss
                loss = torch.sum(yp)
            else:
                loss = torch.sum(inner_loss(yp))

            # For logging purposes
            if innner_ep == 1:  # Store the initial loss
                intial_loss = loss.item()
            if innner_ep == inner_epochs:  # Store the final loss
                final_loss = loss.item()

            CG = create_grad_graph and (inner_epochs - innner_ep < params.unroll_epochs)
            gradients = torch.autograd.grad(
                loss,
                phi_x_batch + phi_y_batch,
                allow_unused=True,
                retain_graph=CG,
                create_graph=CG)
            
            if params.normalize_inner_grad:
                normalized_gradients = [grad / torch.norm(grad) if torch.norm(grad) != 0 else grad for grad in gradients]
            else:
                normalized_gradients = gradients
        

            if params.inner_optimizer != 'sgd':
                phix_phiy = optim.step(phi_x_batch+phi_y_batch, normalized_gradients)
                phi_x_batch = phix_phiy[:len(phi_x_batch)]
                phi_y_batch = phix_phiy[len(phi_x_batch):]
            else:
                learning_rate = [params.inner_lr for i in range(len(phi_x_batch))] if isinstance(params.inner_lr, float) else params.inner_lr
                for i, phi_x in enumerate(phi_x_batch):
                    phi_x_batch[i] = phi_x_batch[i] - learning_rate[i] * normalized_gradients[i]
                for j, phi_y in enumerate(phi_y_batch):
                    phi_y_batch[j] = phi_y_batch[j] - learning_rate[i] * normalized_gradients[j + len(phi_x_batch)] 


            
            if CG and (innner_ep in {1, inner_epochs}) and params.unique_optima_loss:
                output_list.append( yp.clone() if innner_ep == 1 else \
                                   (canonicalizer_model(images, phi_x_batch=phi_x_batch, phi_y_batch=phi_y_batch) if params.surrogate_model else \
                                    model(images, phi_x_list=phi_x_batch, phi_y_list=phi_y_batch, skip_input_augmentation=getattr(params, "skip_input_augmentation", False))).clone())

            for phi_x in phi_x_batch:
                phi_x.grad = None
            for phi_y in phi_y_batch:
                phi_y.grad = None
        if wandb_log:
            with torch.no_grad():
                wandb.log({mode + "inner_loss": intial_loss - final_loss}, commit=True)
                wandb.log({mode + "in_grad_norm":torch.norm( gradients[0]).item()}, commit=True)
    if is_model_training:
        # returning the model to training mode if it was in training mode originally
        model.train()        
    
    if create_grad_graph:
        # return extra output list for unique minima loss during training
        # create grad graph is only true during training
        return phi_x_batch, phi_y_batch, output_list
    else:
        # detach the variables so that they dont take up space
        phi_x_batch = [i.detach() for i in phi_x_batch]
        phi_y_batch = [i.detach() for i in phi_y_batch]
        return phi_x_batch, phi_y_batch
