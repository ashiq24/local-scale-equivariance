from transformers.models.resnet.modeling_resnet import ResNetModel, ResNetStage, ResNetEncoder, ResNetEmbeddings, ResNetPreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
)
from transformers.models.resnet.configuration_resnet import ResNetConfig
from torch import nn, Tensor
from typing import Optional
import torch
from einops import rearrange
from utils.sampling_utils import *
from transformers import AutoImageProcessor
from layers.dense_pediction_head import LinearDensePredHead
from layers.DPT_seg_head import DPTDecoder

class adaResNetEncoder(nn.Module):
    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        self.stages.append(
            ResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        )
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(ResNetStage(config, in_channels, out_channels, depth=depth))

    def forward(
        self, hidden_state: Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        phi_x_list: Optional[torch.FloatTensor] = None,
        phi_y_list: Optional[torch.FloatTensor] = None,
        undo_augmentation: bool = False,
    ) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        # chcek or phi_x_list and phi_y_list should be None together or not
        if phi_x_list is not None and phi_y_list is None or phi_x_list is None and phi_y_list is not None:
            raise ValueError(
                "phi_x_list and phi_y_list should be None together")
        # check or length of phi_x_list and phi_y_list should be equal
        if phi_x_list is not None and len(phi_x_list) != len(phi_y_list):
            raise ValueError(
                "length of phi_x_list and phi_y_list should be equal")
        

        for i, stage_module in enumerate(self.stages):
            if phi_x_list is not None and len(phi_x_list) > i:
                # apply the adtion of parameter phi
                hidden_state = deform(phi_x_list[i], phi_y_list[i], hidden_state)

            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

            if undo_augmentation and phi_x_list is not None and len(phi_x_list) > i:
                hidden_state = inv_deform(phi_x_list[i], phi_y_list[i], hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )

class AdaResNetModel(ResNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedder = ResNetEmbeddings(config)
        self.encoder = adaResNetEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None,
        phi_x_list: Optional[torch.FloatTensor] = None,
        phi_y_list: Optional[torch.FloatTensor] = None,
        skip_input_augment: bool = False,
        undo_augmentation: bool = False
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if skip_input_augment:
            offset = 0
        else:
            if phi_x_list is not None:
                pixel_values = deform(
                    phi_x_list[0],
                    phi_y_list[0],
                    pixel_values)
            offset = 1

        embedding_output = self.embedder(pixel_values)

        if undo_augmentation and phi_x_list is not None and not skip_input_augment:
            embedding_output = inv_deform(
                phi_x_list[0],
                phi_y_list[0],
                embedding_output)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict,
            phi_x_list = phi_x_list[offset:] if phi_x_list is not None else None,
            phi_y_list = phi_y_list[offset:] if phi_y_list is not None else None
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.pooler(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )

class AdaResNetForImageClassification(torch.nn.Module):
    def __init__(self,
                 image_processor_path="microsoft/resnet-50",
                 model_path="microsoft/resnet-50",
                 params={}
                 ):
        super().__init__()
        num_classes = params.num_classes
        use_pretrained = params.use_pretrained
        self.image_processor = AutoImageProcessor.from_pretrained(
            image_processor_path)
        if use_pretrained:
            self.model = AdaResNetModel.from_pretrained(model_path)
        else:
            configuration_resnet = ResNetConfig(num_classes=num_classes)
            self.model = AdaResNetModel(configuration_resnet)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            torch.nn.Linear(self.model.config.hidden_sizes[-1], num_classes)
        )

    def forward(self, x, 
                phi_x_list=None, 
                phi_y_list=None,
                skip_input_augmentation=False,
                undo_augmentation=False,
                return_features=False):
        processed_image = self.image_processor(
            x, return_tensors="pt", do_rescale=False)
        processed_image['pixel_values'] = processed_image['pixel_values'].to(
            x.device)
        
        
        outputs = self.model(
            **processed_image,
            phi_x_list=phi_x_list,
            phi_y_list=phi_y_list,
            skip_input_augment=skip_input_augmentation,
            undo_augmentation=undo_augmentation)
        
        logits = self.classifier(outputs['pooler_output'])

        if return_features:
            return logits
        return logits

# simantic segmentation    
class AdaResNetForDensePrediction(torch.nn.Module):
    def __init__(self,
                image_processor_path="microsoft/resnet-50",
                model_path="microsoft/resnet-50",
                params={},
                ):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(
            image_processor_path)
        
        self.use_pretrained = params.use_pretrained
        self.segmentation_head = params.segmentation_head
        self.dpt_feature_channels = params.dpt_feature_channels
        self.dpt_fusion_hidden_size = params.dpt_fusion_hidden_size
        self.num_classes = params.num_classes

        self.params = params
        if params.use_pretrained:
            self.model = AdaResNetModel.from_pretrained(model_path)
        else:
            configuration_resnet = ResNetConfig(num_classes=self.num_classes)
            self.model = AdaResNetModel(configuration_resnet)
        
        print("hidden sizes", self.model.config.hidden_sizes)
        print("embedding size", self.model.config.embedding_size)
        print("num classes", params.num_classes)

        if self.segmentation_head == 'linear':
            self.decoder = LinearDensePredHead(inchannels=self.model.config.hidden_sizes[-1],
                                            outchannels=params.num_classes)
        elif self.segmentation_head == 'dpt':
            self.decoder = DPTDecoder(in_channels=self.dpt_feature_channels,
                                      fusion_hidden_size=self.dpt_fusion_hidden_size,
                                      num_classes=self.num_classes)
    
    def forward(self,
                x,
                phi_x_list=None, 
                phi_y_list=None,
                skip_input_augmentation=False,
                undo_augmentation=True, ## unlike classification, undo_augmentation is True by default
                return_features=False):
        
        processed_image = self.image_processor(
            x, return_tensors="pt", do_rescale=False)
        
        processed_image['pixel_values'] = processed_image['pixel_values'].to(
            x.device)
        
        if self.segmentation_head == 'linear':
            output_hidden_states = False
        elif self.segmentation_head == 'dpt':
            output_hidden_states = True
        

        outputs = self.model(
            **processed_image,
            phi_x_list=phi_x_list,
            phi_y_list=phi_y_list,
            skip_input_augment=skip_input_augmentation,
            undo_augmentation=undo_augmentation,
            output_hidden_states=output_hidden_states)


        if self.segmentation_head == 'linear':
            logits = self.decoder(outputs['last_hidden_state'],\
                                   output_shape=x.shape[-2:])
        elif self.segmentation_head == 'dpt':
            last_token_output = outputs['hidden_states'][ :len(self.decoder.in_channels)]

            return self.decoder(last_token_output, target_res=x.shape[-2:])
        else:
            raise ValueError("segmentation head not supported")
        return logits
        

        
