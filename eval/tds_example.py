from scipy.stats import entropy
from run_qwen25 import load_model, get_response

### Important: modify transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py to add get_first_token_probs function
#   def get_first_token_probs(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         pixel_values: Optional[torch.Tensor] = None,
#         pixel_values_videos: Optional[torch.FloatTensor] = None,
#         image_grid_thw: Optional[torch.LongTensor] = None,
#         video_grid_thw: Optional[torch.LongTensor] = None,
#         rope_deltas: Optional[torch.LongTensor] = None,
#         cache_position: Optional[torch.Loand image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                     )ngTensor] = None,
#         second_per_grid_ts: Optional[torch.Tensor] = None,
#         token_list: Optional[List] = None,
#     ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if inputs_embeds is None:
#             inputs_embeds = self.model.embed_tokens(input_ids)
#             if pixel_values is not None:
#                 pixel_values = pixel_values.type(self.visual.dtype)
#                 image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#                 n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#                 n_image_features = image_embeds.shape[0]
#                 if n_image_tokens != n_image_features:
#                     raise ValueError(
#                         f"Image features 

#                 mask = input_ids == self.config.image_token_id
#                 mask_unsqueezed = mask.unsqueeze(-1)
#                 mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#                 image_mask = mask_expanded.to(inputs_embeds.device)

#                 image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#                 inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#             if pixel_values_videos is not None:
#                 pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
#                 video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
#                 n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#                 n_video_features = video_embeds.shape[0]
#                 if n_video_tokens != n_video_features:
#                     raise ValueError(
#                         f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                     )

#                 mask = input_ids == self.config.video_token_id
#                 mask_unsqueezed = mask.unsqueeze(-1)
#                 mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#                 video_mask = mask_expanded.to(inputs_embeds.device)

#                 video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#                 inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(inputs_embeds.device)

#         # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
#         if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#             # calculate RoPE index once per generation in the pre-fill stage only
#             if (
#                 (cache_position is not None and cache_position[0] == 0)
#                 or self.rope_deltas is None
#                 or (past_key_values is None or past_key_values.get_seq_length() == 0)
#             ):
#                 position_ids, rope_deltas = self.get_rope_index(
#                     input_ids,
#                     image_grid_thw,
#                     video_grid_thw,
#                     second_per_grid_ts,
#                     attention_mask,
#                 )
#                 self.rope_deltas = rope_deltas
#             # then use the prev pre-calculated rope-deltas to get the correct position ids
#             else:
#                 batch_size, seq_length, _ = inputs_embeds.shape
#                 delta = (
#                     (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
#                     if cache_position is not None
#                     else 0
#                 )
#                 position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#                 position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#                 if cache_position is not None:  # otherwise `deltas` is an int `0`
#                     delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#                 position_ids = position_ids.add(delta)
#                 position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#         outputs = self.model(
#             input_ids=None,
#             position_ids=position_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             cache_position=cache_position,
#         )

#         hidden_states = outputs[0]
#         logits = self.lm_head(hidden_states)
#         if token_list is not None:
#             logits_next_token = logits[:, -1, :]
#             probs = torch.nn.functional.softmax(logits_next_token, dim=-1)
#             token_probs = [probs[0, token_id].item() for token_id in token_list]
#             return token_probs

#         loss = None
#         if labels is not None:
#             # Upcast to float if we need to compute the loss to avoid potential precision issues
#             logits = logits.float()
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return Qwen2_5_VLCausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             rope_deltas=self.rope_deltas,
#         )


def main():
    tokenizer, model, image_processor = load_model('Qwen/Qwen2.5-VL-7B-Instruct')
    
    video_path = 'examples/tempcompass_8307961.mp4'
    query = "What is happening first in the video?\nA. Chatting with a woman while walking\nB. Showing something on his tablet to a woman\nC. They happen at the same time\nAnswer with the option's letter from the given choices directly."
    option_num = 3
    
    p = get_response(model, tokenizer, image_processor, video_path, query, nframes=32, sample_fps=1, frame_process_mode=0, option_num=option_num, return_token_prob=True)
    print('Forward probability: ', p)
    q = get_response(model, tokenizer, image_processor, video_path, query, nframes=32, sample_fps=1, frame_process_mode=1, option_num=option_num, return_token_prob=True)
    print('Reverse probability: ', q)
    kl = entropy(p, q).item()
    print('TDS score for this VQA example using Qwen2.5-VL-7B-Instruct: ', kl)
    
    
if __name__ == "__main__":
    main()