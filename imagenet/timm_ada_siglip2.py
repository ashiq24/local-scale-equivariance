import inspect
from re import DEBUG
from types import MethodType

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from utils.sampling_utils import deform, rerrange_and_scale_tokens

DEBUG = False 

def _clone_and_repeat_phi(phi_list, batch_size, device):
    return [phi.detach().to(device).repeat(batch_size, 1, 1) for phi in phi_list]


def _call_encoder_layer(layer, hidden_states, attention_mask=None, output_attentions=False):
    sig = inspect.signature(layer.forward).parameters
    kwargs = {}
    if "attention_mask" in sig:
        kwargs["attention_mask"] = attention_mask
    if "output_attentions" in sig:
        kwargs["output_attentions"] = output_attentions
    return layer(hidden_states, **kwargs)


def custom_siglip2_vision_forward(
    self,
    pixel_values,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    interpolate_pos_encoding=False,
    attention_mask=None,
    **kwargs,
):
    output_attentions = output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
    )
    return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)

    phi_x_list = self.local_scale_params.param_x_list
    phi_y_list = self.local_scale_params.param_y_list
    batch_size = pixel_values.size(0)

    phi_x_batch = _clone_and_repeat_phi(phi_x_list, batch_size, pixel_values.device)
    phi_y_batch = _clone_and_repeat_phi(phi_y_list, batch_size, pixel_values.device)
    phi_x_batch, phi_y_batch = self.DEM._DEQ(pixel_values, phi_x_batch, phi_y_batch)

    self.tem_x_batch = phi_x_batch
    self.tem_y_batch = phi_y_batch

    aug_index = 0
    unaug_index = 0

    x = pixel_values
    if -1 in self.augment_layer_id:
        if DEBUG:
            print("Applying deformation at input layer")
        x = deform(
            phi_x_batch[aug_index],
            phi_y_batch[aug_index],
            x,
            mode=self.interpolation_mode,
            resolution=self.deform_resolution,
        )
        aug_index += 1

    embedding_sig = inspect.signature(self.embeddings.forward).parameters
    if "interpolate_pos_encoding" in embedding_sig:
        hidden_states = self.embeddings(x, interpolate_pos_encoding=interpolate_pos_encoding)
    else:
        hidden_states = self.embeddings(x)

    num_prefix_tokens = int(getattr(self, "num_prefix_tokens", 0) or 0)
    has_prefix_tokens = True if num_prefix_tokens > 0 else None
    if -1 in self.unaugment_layer_id:
        hidden_states = rerrange_and_scale_tokens(
            phi_x_batch[unaug_index],
            phi_y_batch[unaug_index],
            hidden_states,
            cls_token=has_prefix_tokens,
            num_prefix_tokens=num_prefix_tokens,
            inv_transform=True,
            mode=self.interpolation_mode,
            defom_resolution=self.deform_resolution,
        )
        unaug_index += 1

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    for layer_idx, layer in enumerate(self.encoder.layers):
        
        if DEBUG:
            print(f"Processing layer {layer_idx} feature map size: {hidden_states.shape}")
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if aug_index < len(phi_x_batch) and layer_idx in self.augment_layer_id:
            if DEBUG:
                print(f"Applying deformation at layer {layer_idx}")
            hidden_states = rerrange_and_scale_tokens(
                phi_x_batch[aug_index],
                phi_y_batch[aug_index],
                hidden_states,
                cls_token=has_prefix_tokens,
                num_prefix_tokens=num_prefix_tokens,
                mode=self.interpolation_mode,
                defom_resolution=self.deform_resolution,
            )
            aug_index += 1

        layer_outputs = _call_encoder_layer(
            layer,
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs

        if output_attentions:
            if isinstance(layer_outputs, (tuple, list)) and len(layer_outputs) > 1:
                all_attentions = all_attentions + (layer_outputs[1],)
            else:
                all_attentions = all_attentions + (None,)

        if unaug_index < len(phi_x_batch) and layer_idx in self.unaugment_layer_id:
            hidden_states = rerrange_and_scale_tokens(
                phi_x_batch[unaug_index],
                phi_y_batch[unaug_index],
                hidden_states,
                cls_token=has_prefix_tokens,
                num_prefix_tokens=num_prefix_tokens,
                inv_transform=True,
                mode=self.interpolation_mode,
                defom_resolution=self.deform_resolution,
            )
            unaug_index += 1

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    last_hidden_state = self.post_layernorm(hidden_states) if hasattr(self, "post_layernorm") else hidden_states
    if hasattr(self, "head") and getattr(self, "use_head", True):
        pooler_output = self.head(last_hidden_state)
    elif hasattr(self, "pooler"):
        pooler_output = self.pooler(last_hidden_state)
    else:
        pooler_output = last_hidden_state.mean(dim=1)

    if not return_dict:
        output = (last_hidden_state, pooler_output)
        if output_hidden_states:
            output = output + (encoder_states,)
        if output_attentions:
            output = output + (all_attentions,)
        return output

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooler_output,
        hidden_states=encoder_states,
        attentions=all_attentions,
    )


def _get_siglip2_vision_backbone(model):
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "vision_model"):
        return model.vision_model.vision_model
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "embeddings"):
        return model.vision_model
    raise ValueError(
        "Could not find SigLIP vision backbone. Expected `model.vision_model.vision_model` "
        "or `model.vision_model` with `embeddings` and `encoder`."
    )


def convert_siglip2_to_dem(model, local_scale_params, DEM_model, aug_ids, unaug_ids, deform_res, interpolation_mode="bilinear"):
    vision_backbone = _get_siglip2_vision_backbone(model)

    vision_backbone.local_scale_params = local_scale_params
    vision_backbone.DEM = DEM_model
    vision_backbone.tem_x_batch = None
    vision_backbone.tem_y_batch = None
    vision_backbone.augment_layer_id = aug_ids
    vision_backbone.unaugment_layer_id = unaug_ids
    vision_backbone.interpolation_mode = interpolation_mode
    vision_backbone.deform_resolution = deform_res

    model.tem_x_batch = None
    model.tem_y_batch = None

    vision_backbone.forward = MethodType(custom_siglip2_vision_forward, vision_backbone)
    return model


def convert_siglip2_model(model, adaptation_config, local_scale_params, DEM_model):
    if adaptation_config.do_cannonicalization:
        raise NotImplementedError("Canonicalization mode is not implemented for SigLIP-2.")

    return convert_siglip2_to_dem(
        model,
        local_scale_params,
        DEM_model,
        adaptation_config.augment_layer_id,
        adaptation_config.unaugment_layer_id,
        adaptation_config.deform_resolution,
        adaptation_config.interpolation_mode,
    )
