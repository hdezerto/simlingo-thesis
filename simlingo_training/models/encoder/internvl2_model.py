import os

import torch
from torch import nn
from typing import List, Optional
from transformers import AutoModel

class LingoInternVLModel(nn.Module):
    def __init__(self, variant, *args, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            variant,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        try:
            self.num_embeddings = self.model.language_model.model.embed_tokens.num_embeddings
        except:
            self.num_embeddings = self.model.language_model.vocab_size
        self.use_global_img = None
        self.processor = None

    def extract_frame_features(self, pixel_values: torch.FloatTensor, embed_dim: int) -> torch.FloatTensor:
        """
        Encode all temporal frames, then return features grouped per frame.

        Args:
            pixel_values: [B, T, NP, C, H, W]
            embed_dim: output token size expected by the language model

        Returns:
            Tensor with shape [B, T, P, D], where P is the number of visual
            tokens produced per frame and D == embed_dim.
        """
        BS, T, NP, C, H, W = pixel_values.shape
        pixel_values = pixel_values.reshape(BS * T * NP, C, H, W)
        image_features = self.model.extract_feature(pixel_values)
        return image_features.reshape(BS, T, -1, embed_dim)

    @staticmethod
    def fill_special_token_embeddings(
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int,
        token_embeds: torch.Tensor,
        token_name: str,
    ) -> torch.Tensor:
        for batch_idx in range(input_ids.size(0)):
            token_positions = (input_ids[batch_idx] == token_id).nonzero(as_tuple=False).flatten()
            if token_positions.numel() == 0:
                if token_embeds.size(1) == 0:
                    continue
                raise ValueError(f"Prompt is missing expected {token_name} placeholders.")
            if token_positions.numel() != token_embeds.size(1):
                raise ValueError(
                    f"Expected {token_embeds.size(1)} {token_name} placeholders but found "
                    f"{token_positions.numel()}."
                )
            inputs_embeds[batch_idx, token_positions] = token_embeds[batch_idx]
        return inputs_embeds

    @staticmethod
    def replace_waypoint_placeholders(
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        placeholder_values: Optional[List[dict]],
        wp_encoder: Optional[nn.Module],
    ) -> torch.Tensor:
        if not placeholder_values or wp_encoder is None:
            return inputs_embeds

        wp_encoder_dtype = wp_encoder.mlp[0].weight.dtype
        for batch_idx, placeholder_dict in enumerate(placeholder_values):
            if not placeholder_dict:
                continue
            for token_id, coords_value in placeholder_dict.items():
                token_positions = (input_ids[batch_idx] == token_id).nonzero(as_tuple=False).flatten()
                if token_positions.numel() == 0:
                    continue

                coords = torch.as_tensor(coords_value, device=input_ids.device, dtype=wp_encoder_dtype)
                wp_embeds = wp_encoder(coords.unsqueeze(0)).squeeze(0).to(dtype=inputs_embeds.dtype)

                start = token_positions[0].item()
                end = start + wp_embeds.size(0)
                expected_positions = torch.arange(start, end, device=input_ids.device)
                if token_positions.numel() != wp_embeds.size(0) or not torch.equal(token_positions, expected_positions):
                    raise ValueError(
                        "Placeholder tokens must occupy one contiguous block so their embeddings can be replaced."
                    )
                inputs_embeds[batch_idx, start:end] = wp_embeds

        return inputs_embeds
        
    def replace_placeholder_tokens(
        self,
        adaptor_dict: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        placeholder_values: Optional[List[dict]] = None,
        wp_encoder: Optional[nn.Module] = None,
        temporal_encoder: Optional[nn.Module] = None,
        precomputed_frame_features: Optional[torch.FloatTensor] = None,
    ):
        
        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        TEMP_CONTEXT_TOKEN = '<TEMP_CONTEXT>'
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        temp_context_token_id = self.tokenizer.convert_tokens_to_ids(TEMP_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            # In case image_token_index is not in the embeddings (extra token but embedding don't have it)
            # for_inputs_embeds_ids = input_ids.clone()
            # for_inputs_embeds_ids[(input_ids >= self.num_embeddings)] = 0
            # inputs_embeds = language_model.model.get_input_embeddings()(for_inputs_embeds_ids)
            inputs_embeds = adaptor_dict['language_inputs']
            input_ids = adaptor_dict['language__ids']

            # 2a replace placeholder
            inputs_embeds = self.replace_waypoint_placeholders(
                inputs_embeds,
                input_ids,
                placeholder_values,
                wp_encoder,
            )

            # 2. Merge text and images. Online inference may pass cached frame
            # features so repeated history frames do not need another vision pass.
            if (
                (precomputed_frame_features is not None)
                or (pixel_values is not None and pixel_values.size(0) > 0)
            ) and input_ids.shape[1] != 1:
                _, _, C_embed = inputs_embeds.shape
                if precomputed_frame_features is not None:
                    frame_features = precomputed_frame_features
                else:
                    frame_features = self.extract_frame_features(pixel_values, C_embed)
                current_frame_embeds = frame_features[:, -1].to(dtype=inputs_embeds.dtype)
                inputs_embeds = self.fill_special_token_embeddings(
                    inputs_embeds,
                    input_ids,
                    self.img_context_token_id,
                    current_frame_embeds,
                    IMG_CONTEXT_TOKEN,
                )

                if temporal_encoder is not None:
                    temporal_history_mode = os.environ.get("TEMPORAL_HISTORY_MODE", "real").strip().lower()
                    temporal_history_mode = temporal_history_mode.replace("-", "_")
                    if temporal_history_mode not in {"real", "repeat_current", "zero"}:
                        raise ValueError(
                            "TEMPORAL_HISTORY_MODE must be one of: real, repeat_current, zero. "
                            f"Got: {temporal_history_mode}"
                        )

                    temporal_dtype = next(temporal_encoder.parameters()).dtype
                    if temporal_history_mode == "zero":
                        temporal_embeds = inputs_embeds.new_zeros(
                            frame_features.size(0),
                            temporal_encoder.num_queries,
                            current_frame_embeds.size(-1),
                        )
                    else:
                        if temporal_history_mode == "repeat_current":
                            past_frame_embeds = frame_features[:, -1:].expand(
                                -1,
                                frame_features.size(1) - 1,
                                -1,
                                -1,
                            )
                        else:
                            past_frame_embeds = frame_features[:, :-1]

                        past_frame_embeds = past_frame_embeds.to(dtype=temporal_dtype)
                        if getattr(temporal_encoder, "uses_current_frame", False):
                            current_frame_embeds_for_temporal = frame_features[:, -1].to(dtype=temporal_dtype)
                            temporal_embeds = temporal_encoder(
                                past_frame_embeds,
                                current_frame_tokens=current_frame_embeds_for_temporal,
                            ).to(dtype=inputs_embeds.dtype)
                        else:
                            temporal_embeds = temporal_encoder(past_frame_embeds).to(dtype=inputs_embeds.dtype)
                    inputs_embeds = self.fill_special_token_embeddings(
                        inputs_embeds,
                        input_ids,
                        temp_context_token_id,
                        temporal_embeds,
                        TEMP_CONTEXT_TOKEN,
                    )
            # pixel_values is not None but is empty ---> text only cases
            elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
                # there are no images
                pass

            adaptor_dict['language_inputs'] = inputs_embeds
            start_id = adaptor_dict['perm'][:,0]

            for b, i in enumerate(start_id):
                adaptor_dict['inputs'][b][:len(adaptor_dict['language_inputs'][b])-i] = inputs_embeds[b][i:]

        return adaptor_dict
