from torch import nn
import torch
from utils.sampling_utils import *
import itertools
import torch.nn.functional as F

class CanonicalizeWrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        num_layers,
        num_channels,
        kernel_sizes,
        adapter_coarse_resolution=8,
        task='classification',
        unique_params_limit=15,
        discrete_values=[1, 0.5]
    ):
        '''
        Canonicalization Baseline.
        Wraps the backbone model with generic cannonicalizer.
        Args:
            model: backbone model
            num_layers: number of layers in the cannonicalizer
            num_channels: number of channels in each layer of the cannonicalizer
            kernel_sizes: kernel size for each layer of the cannonicalizer
            adapter_coarse_resolution: The grid size for local scaling operation by monotine function.
            task: classification or segmentation
            unique_params_limit: number of unique parameters of the local scaling to do the cannonicalization 
        '''
        super().__init__()

        self.num_layers = num_layers 
        self.num_channels = num_channels 
        self.kernel_sizes = kernel_sizes
        self.model = model
        self.task = task
        self.unique_params_limit = unique_params_limit
        self.discrete_values = discrete_values
        
        # create model for the cannonicalizer
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                torch.nn.Conv2d(
                    num_channels[i],
                    num_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    padding=(kernel_sizes[i] - 1) // 2,
                )
            )
        # fixed number of parameters for the local scaling operation
        deform_params_x, deform_params_y = self.generate_deform_params(
            adapter_coarse_resolution
        )
        self.register_buffer('deform_params_x',deform_params_x)
        self.register_buffer('deform_params_y',deform_params_y)

    def forward(self, pixel_values, **kwargs):

        best_images, deform_params_x_repeated, deform_params_y_repeated = self.get_canonicalized_images(pixel_values)
        if self.model is None:
            return best_images, deform_params_x_repeated, deform_params_y_repeated
        outputs = self.model(best_images, **kwargs)
        if self.task == 'classification':
            return outputs
 
        return inv_deform(deform_params_x_repeated, deform_params_y_repeated, outputs)


    def get_canonicalized_images(self, pixel_values):
        
        batch_size, num_channels, height, width = pixel_values.size()
        num_possible_warps = self.deform_params_x.size(0)

        pixel_values_repeated = pixel_values.repeat_interleave(num_possible_warps, dim=0)
        deform_params_x_repeated = self.deform_params_x.repeat_interleave(batch_size, dim=0)
        deform_params_y_repeated = self.deform_params_y.repeat_interleave(batch_size, dim=0)

        pixel_values_repeated_deformed = deform(deform_params_x_repeated, deform_params_y_repeated, pixel_values_repeated)
        pixel_values = pixel_values_repeated_deformed.reshape(batch_size * num_possible_warps, num_channels, height, width)

        for i in range(self.num_layers):
            pixel_values = self.conv_layers[i](pixel_values)
            if i < self.num_layers - 1:
                pixel_values = F.relu(pixel_values)
                pixel_values = F.max_pool2d(pixel_values, 4)

        scores = torch.mean(pixel_values, dim=[-2, -1]).reshape(batch_size, num_possible_warps, -1)
        images = pixel_values_repeated_deformed.reshape(batch_size, num_possible_warps, num_channels, height, width)
        
        scores = scores.squeeze(-1)

        batch_size, num_group_elements = scores.shape
        scores_one_hot = F.one_hot(
            torch.argmax(scores, dim=-1), num_group_elements
        ).float()
        scores_soft = F.softmax(scores, dim=-1)

        # Argmax trick
        best_images = torch.sum(
            (scores_one_hot + scores_soft - scores_soft.detach()).reshape(
                batch_size, num_group_elements, 1, 1, 1
            )  # match number of dimensions of images
            * images,
            dim=1,
        )
        
        deform_params_x_repeated = (deform_params_x_repeated.reshape(batch_size, num_possible_warps, *deform_params_x_repeated.shape[-2:]) * scores_one_hot.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        deform_params_y_repeated = (deform_params_y_repeated.reshape(batch_size, num_possible_warps, *deform_params_y_repeated.shape[-2:]) * scores_one_hot.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        
        return best_images, deform_params_x_repeated, deform_params_y_repeated
        

    def generate_deform_params(self, adapter_coarse_resolution):
        middle_column_combinations = list(
        itertools.product(self.discrete_values, repeat=adapter_coarse_resolution)
        )
        params_all = torch.stack(
        [
                torch.stack(
                [       torch.ones(len(middle_column)),
                        torch.ones(len(middle_column)),
                        torch.tensor(middle_column),
                        torch.tensor(middle_column),
                        torch.tensor(middle_column),
                        torch.ones(len(middle_column)),
                        torch.ones(len(middle_column)),
                ],
                dim=1,
                )
                for middle_column in middle_column_combinations
        ]
        )
        unique_params = params_all[:self.unique_params_limit]

        unique_params = list(itertools.product(unique_params, repeat=2))

        params_x = torch.stack([params[0] for params in unique_params])
        params_y = torch.stack([params[1] for params in unique_params])

        return params_x, params_y
