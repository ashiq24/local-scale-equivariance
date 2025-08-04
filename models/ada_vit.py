from typing import Optional, Tuple, Union
from transformers.models.vit import *
from transformers.models.vit.modeling_vit import *
import torch
from transformers import AutoImageProcessor
from utils.sampling_utils import *
from einops import rearrange
from layers.dense_pediction_head import LinearDensePredHead
from layers.DPT_seg_head import DPTDecoder

from transformers.models.vit.configuration_vit import ViTConfig
from utils.sampling_utils import deform, inv_deform, rerrange_and_scale_tokens


class AdaViTEncoder(ViTEncoder):
    def __init__(self, config: ViTConfig) -> None:
        #  (default is "sdpa") does not support double backward. Please change to "eager" if you want to use double backward such as optimization
        # through the transformer output.
        # config._attn_implementation = "eager"
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        phi_x_list: Optional[torch.FloatTensor] = None,
        phi_y_list: Optional[torch.FloatTensor] = None,
        undo_augmentation: bool = False,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # check phi_x_list and phi_y_list should be None together or not
        if (
            phi_x_list is not None
            and phi_y_list is None
            or phi_x_list is None
            and phi_y_list is not None
        ):
            raise ValueError("phi_x_list and phi_y_list should be None together")
        # check length of phi_x_list and phi_y_list should be equal
        if phi_x_list is not None and len(phi_x_list) != len(phi_y_list):
            raise ValueError("length of phi_x_list and phi_y_list should be equal")

        # Iterate over ViT layers
        for i, layer_module in enumerate(self.layer):
            if phi_x_list is not None and len(phi_x_list) > i:
                hidden_states = rerrange_and_scale_tokens(phi_x_list[i], phi_y_list[i], hidden_states, cls_token=0, defom_resolution=self.params.deform_resolution)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, layer_head_mask, output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if undo_augmentation and phi_x_list is not None and len(phi_x_list) > i:
                hidden_states = rerrange_and_scale_tokens(phi_x_list[i], phi_y_list[i], hidden_states, inv_transform=True, cls_token=0, defom_resolution=self.params.deform_resolution)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class AdaViTModel(ViTModel):
    def __init__(
        self,
        config: ViTConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        super().__init__(config, add_pooling_layer, use_mask_token)
        del self.encoder
        self.encoder = AdaViTEncoder(config)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        phi_x_list: Optional[torch.FloatTensor] = None,
        phi_y_list: Optional[torch.FloatTensor] = None,
        skip_input_augment: bool = False,
        undo_augmentation: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        if skip_input_augment:
            offset = 0
        else:
            if phi_x_list is not None:
                pixel_values = deform(
                    phi_x_list[0],
                    phi_y_list[0],
                    pixel_values,
                )
            offset = 1

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        if undo_augmentation and phi_x_list is not None and not skip_input_augment:
            resolution = embedding_output.shape[-2] - 1
            h = int(resolution**0.5)

            # Separate CLS token, only deform image tokens
            cls_token = embedding_output[:, 0, :].unsqueeze(1)
            temp_embd = (
                embedding_output[:, 1:, :]
                .transpose(1, 2)
                .reshape(-1, embedding_output.shape[-1], h, h)
            )
            embedding_output = inv_deform(
                phi_x_list[0], phi_y_list[0], temp_embd)
            embedding_output = embedding_output.reshape(
                -1, embedding_output.shape[1], h * h
            ).transpose(1, 2)

            # Concatenate CLS token back
            embedding_output = torch.cat([cls_token, embedding_output], dim=1)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            phi_x_list=phi_x_list[offset:] if phi_x_list is not None else None,
            phi_y_list=phi_y_list[offset:] if phi_y_list is not None else None,
            undo_augmentation=undo_augmentation,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            head_outputs = (
                (sequence_output, pooled_output)
                if pooled_output is not None
                else (sequence_output,)
            )
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class AdaViTForImageClassification(torch.nn.Module):
    def __init__(
        self,
        image_processor_path="google/vit-base-patch16-224-in21k",
        model_path="google/vit-base-patch16-224-in21k",
        params={},
    ):
        super().__init__()
        num_classes = params.num_classes
        use_pretrained = params.use_pretrained
        self.image_processor = AutoImageProcessor.from_pretrained(
            image_processor_path, use_fast=True
        )
        if use_pretrained:
            self.model = AdaViTModel.from_pretrained(model_path)
        else:
            configuration_vit = ViTConfig(num_classes=num_classes)
            self.model = AdaViTModel(configuration_vit)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_classes)
        self.params = params
        self.model.params = self.params
        self.model.encoder.params = self.params

    def forward(
        self,
        x,
        phi_x_list=None,
        phi_y_list=None,
        skip_input_augmentation=False,
        undo_augmentation=True,
        return_features=False,
    ):
        processed_image = self.image_processor(x, return_tensors="pt", do_rescale=False)
        processed_image["pixel_values"] = processed_image["pixel_values"].to(x.device)

        outputs = self.model(
            **processed_image,
            phi_x_list=phi_x_list,
            phi_y_list=phi_y_list,
            skip_input_augment=skip_input_augmentation,
            undo_augmentation=undo_augmentation,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        return logits

# segmentation module
class AdaViTForDensePrediction(torch.nn.Module):
    def __init__(
        self,
        image_processor_path="google/vit-base-patch16-224-in21k",
        model_path="google/vit-base-patch16-224-in21k",
        params={},
    ):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(
            image_processor_path, use_fast=True
        )

        self.use_pretrained = params.use_pretrained
        self.segmentation_head = params.segmentation_head
        self.dpt_feature_channels = params.dpt_feature_channels
        self.dpt_fusion_hidden_size = params.dpt_fusion_hidden_size
        self.num_classes = params.num_classes

        if self.use_pretrained:
            self.model = AdaViTModel.from_pretrained(model_path)
        else:
            configuration_vit = ViTConfig(num_classes=self.num_classes)
            self.model = AdaViTModel(configuration_vit)

        if self.segmentation_head == "linear":
            self.decoder = LinearDensePredHead(
                inchannels=self.model.config.hidden_size, outchannels=self.num_classes
            )
        elif self.segmentation_head == "dpt":
            self.decoder = DPTDecoder(
                in_channels=self.dpt_feature_channels,
                fusion_hidden_size=self.dpt_fusion_hidden_size,
                num_classes=self.num_classes,
            )
        
        self.params = params
        self.model.params = self.params
        self.model.encoder.params = self.params

    def forward(
        self,
        x,
        phi_x_list=None,
        phi_y_list=None,
        skip_input_augmentation=False,
        undo_augmentation=True,
        return_features=False,
    ):
        processed_image = self.image_processor(x, return_tensors="pt", do_rescale=False)
        processed_image["pixel_values"] = processed_image["pixel_values"].to(x.device)

        if self.segmentation_head == "linear":
            output_hidden_states = False
        elif self.segmentation_head == "dpt":
            output_hidden_states = True

        outputs = self.model(
            **processed_image,
            phi_x_list=phi_x_list,
            phi_y_list=phi_y_list,
            skip_input_augment=skip_input_augmentation,
            undo_augmentation=undo_augmentation,
            output_hidden_states=output_hidden_states,
        )

        if self.segmentation_head == "linear":
            last_token_output = rearrange(
                outputs["last_hidden_state"][:, 1:, :], "b (h w) c -> b c h w", 
                h=int(outputs["last_hidden_state"].shape[-2]**0.5), 
                w=int(outputs["last_hidden_state"].shape[-2]**0.5)
            )
            logits = self.decoder(last_token_output, output_shape=x.shape[-2:])

            return logits
        elif self.segmentation_head == "dpt":
            last_token_output = tuple(
                torch.stack(outputs["hidden_states"][ :len(self.decoder.in_channels)])[:, :, 1:, :]
            )
            return self.decoder(last_token_output, target_res=x.shape[-2:])
        else:
            raise ValueError("Invalid segmentation head type")
