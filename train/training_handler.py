import torch
import math
from utils.sampling_utils import *
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F

######
# Helper functions for training
######

def entropy(x):
    '''
    Compute the entropy of the input x
    '''
    normalizing_factor = 1
    if len(x.shape) == 2: # (batch, num_classes)
        x = torch.softmax(x, dim=-1)
        return -torch.sum(x * torch.log(x + 1e-8), dim=-1)/normalizing_factor
    elif len(x.shape) == 3: # (batch, num_classes, H)
        # entropy loss
        x = torch.softmax(x, dim=1)
        normalizing_factor = x.shape[-1]
        return -torch.sum(x * torch.log(x + 1e-8), dim=1)/normalizing_factor
    elif len(x.shape) == 4: # (batch, num_classes, H, W)
        x = torch.softmax(x, dim=1)
        normalizing_factor = x.shape[-1] * x.shape[-2]
        return -torch.sum(x * torch.log(x + 1e-8), dim=1)/normalizing_factor
    else:
        raise ValueError(f'x shape {x.shape} not supported')

def symmetric_kl_div(res_p_soft, yp_soft):
    '''
    Compute the symmetric KL divergence between the two distributions
    '''
    # Compute the midpoint distribution
    M = 0.5 * (res_p_soft + yp_soft)

    # Compute KL(P || M) and KL(Q || M)
    kl_p_m = F.kl_div(M.log(), yp_soft, reduction='batchmean')  # KL(P || M)
    kl_q_m = F.kl_div(M.log(), res_p_soft, reduction='batchmean')  # KL(Q || M)

    # JS divergence is the average of the two
    js_div = 0.5 * (kl_p_m + kl_q_m)
    return js_div

def renyi_entropy(x, alpha=2):
    '''
    Compute the Renyi entropy of the input x
    '''
    assert alpha != 1
    x = torch.softmax(x, dim=-1)
    return torch.sum(x**alpha, dim=-1) / (1 - alpha)


def get_inner_loss(params):
    '''
    Get the inner loss function
    '''
    if params.inner_loss == "renyi":
        return renyi_entropy
    return entropy

###########################
class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup and cosine annealing.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer
        warmup_epochs (int): Number of warmup epochs
        total_epochs (int): Total number of training epochs
        eta_min (float): Minimum learning rate (default: 0)
        last_epoch (int): The index of last epoch (default: -1)
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch / self.warmup_epochs) 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            current_epoch = self.last_epoch - self.warmup_epochs
            cosine_epochs = self.total_epochs - self.warmup_epochs
            
            # Cosine decay calculation
            cosine_factor = (1 + math.cos(math.pi * current_epoch / cosine_epochs)) / 2
            return [self.eta_min + (base_lr - self.eta_min) * cosine_factor 
                    for base_lr in self.base_lrs]



def get_scheduler(params, optim):
    '''
    Returns the scheduler depending on the parameters
    '''
    if params.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR
    elif params.scheduler == 'c':
        scheduler = torch.optim.lr_scheduler.CyclicLR
    elif params.scheduler == 'rdp':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    else:
        raise ValueError(f'Scheduler {params.scheduler} not found')
    return scheduler(optim, **params.scheduler_kwargs)


def get_optimizer(params, model, lr, kwargs):
    '''
    Returns the optimizer depending on the parameters
    '''
    if params.optimizer == 'adam':
        optimizer = torch.optim.AdamW
    elif params.optimizer == 'sgd':
        optimizer = torch.optim.SGD
    elif params.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop
    else:
        raise ValueError(f'Optimizer {params.optimizer} not found')
    return optimizer(model.parameters(), lr=lr, **kwargs)


def get_loss(params):
    '''
    Returns the loss function depending on the task
    '''
    if params.criterion == "cross_entropy":
        if getattr(params, 'task', 'classification') == 'segmentation':
            weights = torch.ones(params.num_classes).to(params.device)
            weights[0] = weights[0]/10
            loss = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss = torch.nn.CrossEntropyLoss()
    elif params.criterion == "mse":
        loss = torch.nn.MSELoss()
    else:
        raise ValueError(f'Loss {params.loss} not found')
    return loss

def random_data_augmentation(images, labels, params):
    '''
    apply random local scale augmentation to the images and labels
    For classification, only images are augmented
    For segmentation, both images and labels are augmented
    
    args:
        images: input images
        labels: input labels
        params: parameters (config)

    returns:
        images: augmented images, augmented labels, augmentation parameters x, augmentation parameters y
    '''
    device = getattr(params, 'device', None)
    if device is not None: images = images.to(device)
    resolution = params.warping_resolution
    strength = params.warping_strength
    phi_x_aug = torch.rand(
        images.size(0), resolution, resolution - 1).to(device)  * strength
    phi_y_aug = torch.rand(
        images.size(0), resolution, resolution - 1).to(device)  * strength
    
    images = deform(phi_x_aug, phi_y_aug, images)
    if getattr(params, 'task', "classification") == "segmentation":
        labels = labels.float()
        labels = labels.to(device)
        labels = labels.unsqueeze(1)
        labels = deform(phi_x_aug, phi_y_aug, labels, mode='nearest')
        labels = labels.squeeze(1)
        labels = torch.ceil(labels).long()
    
    return images, labels, phi_x_aug, phi_y_aug

def get_equivariance_loss(x,
                         y, 
                         model,
                         surrogate_model,
                         params,
                         yp,
                         phi_x_list,
                         phi_y_list,
                         inner_optimization
                         ):
    '''
    Compute the equivariance loss for local scaling operation, i.e., it returns
    distance(M(x) - M(Local_scale(x)))
    distance can be KL divergence, MSE, etc.

    args:
        x: input images
        y: input labels
        model: model to be used for forward pass
        surrogate_model: model to be used for inner optimization
        params: parameters (config)
    '''
    with torch.no_grad():
        x_aug, _, aug_x, aug_y = random_data_augmentation(x.clone().detach(), y.clone().detach(), params)
    if params.do_adaptation:
        phi_x_batch, phi_y_batch, _ = inner_optimization(model=model,
                                                                canonicalizer_model=surrogate_model,
                                                                images=x_aug,
                                                                params=params,
                                                                phi_x_list=phi_x_list,
                                                                phi_y_list=phi_y_list,
                                                                create_grad_graph=True,
                                                                wandb_log=False)
    else:
        phi_x_batch = None
        phi_y_batch = None
    res_p = model(x_aug,
                  phi_x_list=phi_x_batch,
                  phi_y_list=phi_y_batch, 
                  skip_input_augmentation=getattr(
                                          params, "skip_input_augmentation", False))
    error = calculate_eq_loss(yp, res_p, aug_x, aug_y, params)

    return error

def calculate_eq_loss(yp, res_p, aug_x, aug_y, params):
    '''
    yp: output of the model
    res_p: output of the model with augmented input
    aug_x: input augmentation parameters x
    aug_y: input augmentation parameters y
    params: parameters
    '''
    loss_type = getattr(params, 'eqloss_type', 'KL')
    if params.task == 'classification':
        if loss_type == 'MSE':
            error = F.mse_loss(res_p, yp)
        else:
            res_p_soft = torch.softmax(res_p, dim=1)
            yp_soft = torch.softmax(yp, dim=1)
            # calculate KL divergencve between yp and res_p
            if loss_type== "KL":
                error = torch.mean(torch.sum(yp_soft * torch.log(yp_soft + 1e-8) - yp_soft * torch.log(res_p_soft + 1e-8), dim=-1))
            else:
                error = symmetric_kl_div(res_p, yp)
            
    elif params.task == 'segmentation':
        aug_res_soft = deform(aug_x, aug_y, yp)
        if loss_type == 'MSE':
            error = F.mse_loss(aug_res_soft, res_p)
        else:
            aug_res_soft = torch.softmax(aug_res_soft, dim=1)
            res_p_soft = torch.softmax(res_p, dim=1)
            
            num_classes = res_p_soft.shape[1]
            # reshape
            aug_res_soft = aug_res_soft.permute(0, 2, 3, 1).reshape(-1, num_classes)
            res_p_soft = res_p_soft.permute(0, 2, 3, 1).reshape(-1, num_classes)
            # claculate the KL divergence between res_p and aug_res
            if loss_type== "KL":
                error = torch.mean(torch.sum(aug_res_soft * torch.log(aug_res_soft + 1e-8) - aug_res_soft * torch.log(res_p_soft + 1e-8), dim=-1))
            else:
                error = symmetric_kl_div(res_p, aug_res_soft)

    return error

def weight_task_loss(loss, params, current_model_under_training):
    if current_model_under_training == 'surrogate':
        return params.surrogate_task_loss_weight * loss
    else:
        return loss
    
def weight_equivariance_loss(loss, params, current_model_under_training):
    if current_model_under_training == 'surrogate':
        return params.surrogate_equivariance_loss_weight * loss
    else:
        return params.equivariance_loss_weight*loss   

def get_unique_optima_loss(out_list, phi_x_batch, phi_y_batch, Batch_Size, params, epoch, return_sep_loss=False):
    '''
    The following loss enforces the model to learn the unique optima of different local scaling operations.
    This enforces to have a unique optima on the orbit of the local scaling group.
    '''
    phi_diff_loss = 0
    phi_prior_loss = 0
    for i,j in zip(phi_x_batch, phi_y_batch):
        coordinates = get_coarse_adaptive_grid(i, j)
        uni_grid = get_coarse_adaptive_grid(torch.ones(1, i.shape[-2], i.shape[-1]),
                                            torch.ones(1, j.shape[-2], j.shape[-1])).repeat(2*Batch_Size,1, 1, 1).to(i.device)
        coordinates = coordinates.reshape(2*Batch_Size, -1) 
        uni_grid = uni_grid.reshape(2*Batch_Size, -1)


        # increasing the distance between the augmentations on augmented vs unaugmented
        phi_diff_weight_loss = min(params.phi_diff_loss_weight + epoch * params.phi_diff_increment_per_epoch,  params.phi_diff_loss_weight_max)
        diff_norm = torch.norm(coordinates[:Batch_Size] - coordinates[Batch_Size:], dim=-1)**2
        diff_norm = -1*torch.clamp(diff_norm, max=0.01)

        phi_diff_loss += torch.mean(diff_norm)
        # regularization so that augmentation is not too much
        phi_prior_loss += torch.mean((coordinates - uni_grid)**2)

    if params.deep_equlibrium_mode:
        if return_sep_loss: 
            return phi_diff_weight_loss * phi_diff_loss +  params.phi_prior_loss_weight*phi_prior_loss, phi_diff_loss, phi_prior_loss
        return getattr(params,'phi_loss_weight', 0.1) * (phi_diff_weight_loss * phi_diff_loss +  params.phi_prior_loss_weight*phi_prior_loss)
    
    # return the following loss, when not useing deep equlibrium mode
    initial_can_val = out_list[0]
    final_can_val = out_list[1]
    init_unaug, init_aug = initial_can_val[:Batch_Size], initial_can_val[Batch_Size:]
    final_unaug, final_aug = final_can_val[:Batch_Size], final_can_val[Batch_Size:]
    
    mean_unique_loss = getattr(params,'unique_optima_loss_weight', 0.5)*\
                        (-2.0*torch.mean((init_unaug - init_aug)**2) + torch.mean((final_unaug - final_aug)**2)) 
    return mean_unique_loss

class OptimizerHandler:
    '''
    Handles the optimizer and scheduler for the model and surrogate model
    '''
    def __init__(self,
                 params):
        self.params = params
        self.current_model_under_training = None # will alternate between 'surrogate' and 'model' for alternate mode

    
    def train(self, model, surrograte_model, ep):
        if self.params.train_surrogate_only:
            # only train the surrogate model
            surrograte_model.train()
            model.eval()
            self.current_model_under_training = 'surrogate'
        elif self.params.alternate_training:
            # fixing which model to train
            if self.current_model_under_training is None:
                # start with the surrogate model
                self.current_model_under_training = 'surrogate'
            elif ep > self.params.alternate_interval:
                self.current_model_under_training = 'both' # training both models

            if self.current_model_under_training == 'surrogate':
                surrograte_model.train()
                model.eval()
            elif self.current_model_under_training == 'model':
                model.train()
                surrograte_model.eval()
            else:
                model.train()
                surrograte_model.train()
        else:
            model.train()
            if surrograte_model is not None: surrograte_model.train()
            self.current_model_under_training = None # this parameter will be ignored

        print(f"********Training {self.current_model_under_training} model********")

    def zero_grad(self, outer_optim, surrogate_optim, ep):
        """
        outter_optim: optimizer for the model
        surrogate_optim: optimizer for the surrogate model
        """
        if self.current_model_under_training == 'surrogate':
            surrogate_optim.zero_grad()
        elif self.current_model_under_training == 'model':
            outer_optim.zero_grad()
        else:
            if outer_optim is not None: outer_optim.zero_grad()
            if surrogate_optim is not None: surrogate_optim.zero_grad()
    
    def step(self, outer_optim, surrogate_optim, ep):
        if self.current_model_under_training == 'surrogate':
            surrogate_optim.step()
        elif self.current_model_under_training == 'model':
            outer_optim.step()
        else:
            if outer_optim is not None: outer_optim.step()
            if surrogate_optim is not None: surrogate_optim.step()
    
    def scheduler_step(self,
                       outer_scheduler,
                       surrogate_scheduler,
                       ep,
                       loss):
        if self.current_model_under_training == 'surrogate':
            surrogate_scheduler.step(loss) if self.params.scheduler == 'rdp' else surrogate_scheduler.step()
        elif self.current_model_under_training == 'model':
            outer_scheduler.step(loss) if self.params.scheduler == 'rdp' else outer_scheduler.step()
        else:
            if outer_scheduler is not None: outer_scheduler.step(loss) if self.params.scheduler == 'rdp' else outer_scheduler.step()
            if surrogate_scheduler is not None: surrogate_scheduler.step(loss) if self.params.scheduler == 'rdp' else surrogate_scheduler.step()
    
    def clip_grad(self, model, surrogate_model):
        if self.current_model_under_training == 'surrogate':
            torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), self.params.surrogate_gradient_clip_value)
        elif self.current_model_under_training == 'model':
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.params.gradient_clip_value)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.params.gradient_clip_value)
            if surrogate_model  is not None:
                torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), self.params.surrogate_gradient_clip_value)

            