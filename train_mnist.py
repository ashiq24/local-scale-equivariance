from YParams import YParams
import os
import wandb
import argparse
import sys
import torch
import numpy as np
from train.trainer import *
from utils.core_utils import *
from models.model_handler import get_model, get_surrogate_model
from evaluation.evaluations import *
from dataloader.dataset_handler import get_data_module, get_test_data_module
from layers.adapter_params import PerlayerAdapterParams
from models.adapter_wrapper import AdapterWrapper
from pathlib import Path
import random
import traceback

def  string_to_bool(s):
    if s == 'True' or s == 'true':
        return True
    else:
        return False
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config_file",
            nargs="?",
            default="config_1.yaml",
            type=str)
        parser.add_argument(
            "--config",
            nargs="?",
            default="base_config",
            type=str)
        parser.add_argument("--epochs", nargs="?", default=None, type=int)
        parser.add_argument("--outer_lr", nargs="?", default=None, type=float)
        parser.add_argument("--random_seed", nargs="?", default=42, type=int)
        parser.add_argument(
            "--adapter_coarse_resolution",
            nargs="?",
            default=None,
            type=list)
        parser.add_argument("--inner_lr", nargs="?", default=None, type=float)
        parser.add_argument(
            "--inner_epochs",
            nargs="?",
            default=None,
            type=int)
        parser.add_argument(
            "--unroll_epochs",
            nargs="?",
            default=None,
            type=int)
        parser.add_argument("--batch_size", nargs="?", default=None, type=int)
        parser.add_argument(
            "--normalize_inner_grad",
            nargs="?",
            default=None,
            type=str)
        parser.add_argument(
            "--learn_phi_intialization",
            nargs="?",
            default=None,
            type=str)
        parser.add_argument("--load_pretrained", nargs="?", default=None, type=str)
        parser.add_argument("--sep_optimization", nargs="?", default=None, type=str)
        parser.add_argument("--test_inner_epochs", nargs="?", default=None, type=int)
        parser.add_argument("--multiscale_fine_tune", nargs="?", default=None, type=str)
        parser.add_argument("--train_augmentation_limit", nargs="?", default=None, type=int)

        parsed_args = parser.parse_args()

        config = parsed_args.config
        print("Loading config", config)
        params = YParams('./config/'+parsed_args.config_file, config, print_params=True)
        Path(params.save_model_path).mkdir(parents=True, exist_ok=True)

        if parsed_args.random_seed is not None:
            params.random_seed = parsed_args.random_seed
            print("Overriding random seed to", params.random_seed)
        if parsed_args.epochs is not None:
            params.epochs = parsed_args.epochs
            print("Overriding epochs to", params.epochs)
        if parsed_args.outer_lr is not None:
            params.outer_lr = parsed_args.outer_lr
            print("Overriding outer_lr to", params.outer_lr)
        if parsed_args.adapter_coarse_resolution is not None:
            params.adapter_coarse_resolution = parsed_args.adapter_coarse_resolution
            print("Overriding adapter_coarse_resolution to",params.adapter_coarse_resolution)
        if parsed_args.inner_lr is not None:
            params.inner_lr = parsed_args.inner_lr
            params.test_inner_lr = parsed_args.inner_lr
            print("Overriding inner_lr to", params.inner_lr)
        if parsed_args.inner_epochs is not None:
            params.inner_epochs = parsed_args.inner_epochs
            params.test_inner_epochs = parsed_args.inner_epochs
            print("Overriding inner_epochs to", params.inner_epochs)
        if parsed_args.unroll_epochs is not None:
            params.unroll_epochs = parsed_args.unroll_epochs
            print("Overriding unroll_epochs to", params.unroll_epochs)
        if parsed_args.batch_size is not None:
            params.batch_size = parsed_args.batch_size
            print("Overriding batch_size to", params.batch_size)
        if parsed_args.normalize_inner_grad is not None:
            params.normalize_inner_grad = parsed_args.normalize_inner_grad
            print("Overriding normalize_inner_grad to",params.normalize_inner_grad)
        if parsed_args.learn_phi_intialization is not None:
            params.learn_phi_intialization = string_to_bool(parsed_args.learn_phi_intialization)
            print("Overriding learn_phi_intialization to",params.learn_phi_intialization)
        if parsed_args.load_pretrained is not None:
            params.load_pretrained = string_to_bool(parsed_args.load_pretrained)
            print("Overriding load_pretrained to",params.load_pretrained)
        if parsed_args.test_inner_epochs is not None:
            params.test_inner_epochs = parsed_args.test_inner_epochs
            print("Overriding test_inner_epochs to",params.test_inner_epochs)
        if parsed_args.sep_optimization is not None:
            params.sep_optimization = string_to_bool(parsed_args.sep_optimization)
            print("Overriding sep_optimization to",params.sep_optimization)
        if parsed_args.multiscale_fine_tune is not None:
            params.multiscale_fine_tune = string_to_bool(parsed_args.multiscale_fine_tune)
            print("Overriding multiscale_fine_tune to",params.multiscale_fine_tune)
        if parsed_args.train_augmentation_limit is not None:
            params.train_augmentation_limit = parsed_args.train_augmentation_limit

        if params.deep_equlibrium_mode:
            params.surrogate_model = True
        

        torch.manual_seed(params.random_seed)
        random.seed(params.random_seed)
        np.random.seed(params.random_seed)

        params.config = config

        # Set up WandB logging
        params.wandb_name = config
        params.wandb_group = params.model
        if params.wandb_log:
            wandb.login(key=get_wandb_api_key())
            # Initialize API client
            api = wandb.Api()

            # Fetch all projects under your account
            projects = api.projects()

            # Extract project names
            project_names = [project.name for project in projects]

            # Get project count
            project_count = len(project_names)

            print(f"Total Projects: {project_count}")
            print("Project Names:", project_names)
            wandb.init(config=params, name=params.wandb_name, project=params.wandb_project)

        model = get_model(params)
        if params.surrogate_model:
            surrogate_model = get_surrogate_model(params)
        else:
            surrogate_model = None

        data_module = get_data_module(params)
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        test_dataloader = data_module.test_dataloader()
        test_dataset = data_module.test_data

        # print Data stat
        print("Train data size", len(train_dataloader.dataset))
        if val_dataloader is not None: print("Val data size", len(val_dataloader.dataset))
        print("Test data size", len(test_dataloader.dataset))
        
        if params.data_set == 'mnist':
            test_data_module = get_test_data_module(params)
            test_dataloader = test_data_module.test_dataloader()

        # get adaptiver parameters
        if params.do_adaptation:
            adapter_parameters = PerlayerAdapterParams(
                num_layers=params.num_phi_layers,
                adapter_coarse_resolution=params.adapter_coarse_resolution)
        else:
            adapter_parameters = None

        if getattr(params, "evaluate_trained_model", False):
            print("loading pretrained model")
            model.load_state_dict(torch.load(params.model_weights_path, weights_only=True))
            model.eval()
            phi_x_list = torch.load(params.phi_x_list_path, weights_only=True) if params.do_adaptation else None
            phi_y_list = torch.load(params.phi_y_list_path, weights_only=True) if params.do_adaptation else None
            if params.surrogate_model:
                surrogate_model.load_state_dict(torch.load(params.surrogate_weights_path, weights_only=True))
        else:
            if getattr(params, "fine_tune", False):
                if params.do_cannonicalization:
                    model.model.load_state_dict(torch.load(params.fine_tune_model_path, weights_only=True))
                else:
                    model.load_state_dict(torch.load(params.fine_tune_model_path, weights_only=True))
                if getattr(params, "multiscale_fine_tune", False):
                    # for segmentation val dataloader will contain more scales for finetuning
                    train_dataloader = val_dataloader
                    
            if params.do_pre_training_eval:
                evaluate_all_metric(model,  None, None,test_dataloader, test_dataset, params)
            bi_level_train(
            model=model,
            surrogate_model=surrogate_model,
            phi_x_list=adapter_parameters.param_x_list if adapter_parameters is not None else None,
            phi_y_list=adapter_parameters.param_y_list if adapter_parameters is not None else None,
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            test_dataset=test_dataset,
            params=params,
            )

        wrapped_model = AdapterWrapper(
            model=model,
            surrogate_model=surrogate_model,
            phi_x_list=adapter_parameters.param_x_list if adapter_parameters is not None else None,
            phi_y_list=adapter_parameters.param_y_list if adapter_parameters is not None else None,
            params=params)

        if getattr(params, "load_pretrained", False):
            evaluate_model(
                wrapped_model,
                test_dataloader,
                params,
            )

        if params.calculate_warping_equiv:
            warping_equivariance_error(
                wrapped_model, test_dataloader, params)

        if params.data_set == 'mnist':
            local_scale_equivariance_testing(
                wrapped_model, test_dataloader, params)

        if params.task == 'segmentation':
            consistency_test_segmentation(
                wrapped_model, test_dataset, params) 
    except Exception as e:
        traceback.print_exc()

    finally:
        if params.wandb_log:
            wandb.finish()
