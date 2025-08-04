from typing import Optional, Tuple, Union
from transformers.models.swin import *
from transformers.models.swin.modeling_swin import *
import torch
from transformers import AutoImageProcessor
from utils.sampling_utils import *
from einops import rearrange
from layers.dense_pediction_head import LinearDensePredHead
from layers.DPT_seg_head import DPTDecoder

class AdaSwinEncoder(SwinEncoder):
    def __init__(self, config, grid_size):
        super().__init__(config, grid_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        phi_x_list: Optional[torch.FloatTensor] = None,
        phi_y_list: Optional[torch.FloatTensor] = None,
        undo_augmentation: bool = False,
    ) -> Union[Tuple, SwinEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # chcek or phi_x_list and phi_y_list should be None together or not
        if phi_x_list is not None and phi_y_list is None or phi_x_list is None and phi_y_list is not None:
            raise ValueError(
                "phi_x_list and phi_y_list should be None together")
        # check or length of phi_x_list and phi_y_list should be equal
        if phi_x_list is not None and len(phi_x_list) != len(phi_y_list):
            raise ValueError(
                "length of phi_x_list and phi_y_list should be equal")

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(
                batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):

            if phi_x_list is not None and len(phi_x_list) > i:
                hidden_states = rerrange_and_scale_tokens(phi_x_list[i], phi_y_list[i], hidden_states, defom_resolution=self.params.deform_resolution)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    always_partition)

            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            if undo_augmentation and phi_x_list is not None and len(phi_x_list) > i:
                hidden_states = rerrange_and_scale_tokens(phi_x_list[i], phi_y_list[i], hidden_states, inv_transform=True, defom_resolution=self.params.deform_resolution)

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size, *
                    (output_dimensions[0], output_dimensions[1]), hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(
                    0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(
                    batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(
                    0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions] if v is not None)

        return SwinEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class AdaSwinModel(SwinModel):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__(
            config,
            add_pooling_layer=add_pooling_layer,
            use_mask_token=use_mask_token)
        # deleting previous encoder
        del self.encoder
        self.encoder = AdaSwinEncoder(config, self.embeddings.patch_grid)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
        phi_x_list: Optional[torch.FloatTensor] = None,
        phi_y_list: Optional[torch.FloatTensor] = None,
        skip_input_augment: bool = False,
        undo_augmentation: bool = False,
    ) -> Union[Tuple, SwinModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x
        # num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))
        if skip_input_augment:
            offset = 0
        else:
            if phi_x_list is not None:
                pixel_values = deform(
                    phi_x_list[0],
                    phi_y_list[0],
                    pixel_values)
            offset = 1

        embedding_output, input_dimensions = self.embeddings(
            pixel_values, 
            bool_masked_pos=bool_masked_pos, 
            interpolate_pos_encoding=interpolate_pos_encoding)
        
        if undo_augmentation and phi_x_list is not None and not skip_input_augment:
            resolution = embedding_output.shape[-2]
            h = int(resolution**0.5)
            temp_embd = embedding_output.transpose(
                1, 2).reshape(-1, embedding_output.shape[-1], h, h)
            embedding_output = inv_deform(phi_x_list[0], phi_y_list[0], temp_embd)
            embedding_output = embedding_output.reshape(
                -1, embedding_output.shape[1], h * h).transpose(1, 2)

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            phi_x_list=phi_x_list[offset:] if phi_x_list is not None else None,
            phi_y_list=phi_y_list[offset:] if phi_y_list is not None else None,
            undo_augmentation=undo_augmentation
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]

            return output

        return SwinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )


class AdaSwinForImageClassification(torch.nn.Module):
    def __init__(self,

                 image_processor_path="microsoft/swin-tiny-patch4-window7-224",
                 model_path="microsoft/swin-tiny-patch4-window7-224",
                 params={}
                 ):
        super().__init__()
        num_classes = params.num_classes
        use_pretrained = params.use_pretrained
        self.image_processor = AutoImageProcessor.from_pretrained(
            image_processor_path)
        if use_pretrained:
            self.model = AdaSwinModel.from_pretrained(model_path)
        else:
            configuration_swin = SwinConfig(num_classes=num_classes)
            self.model = AdaSwinModel(configuration_swin)
        self.classifier = torch.nn.Linear(
            self.model.config.hidden_size, num_classes)
        self.params = params
        self.model.params = self.params
        self.model.encoder.params = self.params


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

# segmentation module

class AdaSwinForDensePrediction(torch.nn.Module):
    def __init__(self,
                 image_processor_path="microsoft/swin-tiny-patch4-window7-224",
                 model_path="microsoft/swin-tiny-patch4-window7-224",
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
        

        if self.use_pretrained:
            self.model = AdaSwinModel.from_pretrained(model_path)
        else:
            configuration_swin = SwinConfig(num_classes=self.num_classes)
            self.model = AdaSwinModel(configuration_swin)
        
        if self.segmentation_head == 'linear':
            self.decoder = LinearDensePredHead(inchannels=self.model.config.hidden_size,
                                            outchannels=self.num_classes)
        elif self.segmentation_head == 'dpt':
            self.decoder = DPTDecoder(in_channels=self.dpt_feature_channels,
                                      fusion_hidden_size=self.dpt_fusion_hidden_size,
                                      num_classes=self.num_classes)
        
        self.params = params
        self.model.params = self.params
        self.model.encoder.params = self.params

    def forward(self, 
                x, 
                phi_x_list=None, 
                phi_y_list=None,
                skip_input_augmentation=False,
                undo_augmentation=True, 
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
            last_token_output = outputs['last_hidden_state']
            # reshape the output
            n = last_token_output.shape[-2]
            H = int(n**0.5)
            W = int(n**0.5)
            last_token_output = rearrange(last_token_output, 'b (h w) c -> b c h w', h=H, w=W)
            logits = self.decoder(last_token_output, output_shape=x.shape[-2:])

            if return_features:
                return logits
            return logits
        elif self.segmentation_head == 'dpt':
            last_token_output = outputs['hidden_states'][ :len(self.decoder.in_channels)]

            return self.decoder(last_token_output, target_res=x.shape[-2:])
        else:
            raise ValueError('Invalid segmentation head type')
