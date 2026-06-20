"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    id = 0
    vocab = {}
    for _, token in enumerate (specials) :
        vocab[token] = id 
        id+=1

    for  _, sentence in enumerate(sentences) :
        tokens = sentence.split()
        for  _, token in enumerate(tokens):
            if token not in vocab :
                vocab[token] = id
                id+=1 

    return vocab

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    id_to_token = {}
    for key, val in token_to_id.items() :
        id_to_token[val] = key
    return id_to_token

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    encoded_sentence = []
    for _, token in enumerate(sentence.split()) :
        if token in token_to_id :
            encoded_sentence.append(token_to_id[token])
        else :
            encoded_sentence.append(token_to_id[unk_token])
    
    return encoded_sentence

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    sentence = []
    for _, id in enumerate(ids) :
        sentence.append(id_to_token[id])
    return sentence

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    padded_list = []
    for i in range(0,max_len) :
        if i >= len(ids) :
            padded_list.append(pad_id)
        else :
            padded_list.append(ids[i])
    return padded_list

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    batch = torch.tensor(padded_sequences, dtype=torch.long)
    return batch

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

# why do we do this ?
'''
We intialize embeddings with small values. So if we add positional
embeddings directly, we will lose the meaning of actual embedding 
since postional embeddings range btw +1 and -1 and values are 
initalized in order of 0.1. 

In actual scenario d is large like 512 or 1024 which scales
the embedding values to 2. something . Now positional encoding
will act like sub embedding and actual embedding values wont get
washed or lose their meaning.
'''
def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    i = torch.arange(0, d_model, 2, dtype = torch.float32)
    div_term = torch.exp(i* -(torch.log(torch.tensor(10000)))/d_model)
    return div_term

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    x = torch.arange(max_len, dtype = torch.float32)
    return x.reshape(-1, 1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    pe[:, 0::2] = torch.sin(position * div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    pe[:, 1::2] = torch.cos(position*div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    div_term = torch.arange(0, d_model, 2, dtype = torch.float32)
    div_term = torch.exp(div_term*-(torch.log(torch.tensor(10000)))/d_model)
    position = torch.arange(max_len, dtype = torch.float32)
    position = position.reshape (-1, 1)
    pe[:, 0::2] = torch.sin(position*div_term)
    pe[:, 1::2] = torch.cos(position*div_term)
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    l  = embedded_batch.shape[1]
    return embedded_batch + positional_encoding[:l]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    padding_mask = torch.where(token_ids==pad_id, False, True)

    return padding_mask.unsqueeze(1).unsqueeze(1)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    row_index = torch.arange(seq_len).unsqueeze(1)
    col_index = torch.arange(seq_len).unsqueeze(0)
    return (row_index>=col_index).unsqueeze(0).unsqueeze(0)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    return query@key.transpose(-2, -1)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
   return scores/math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    return torch.where(mask, scores, -float('inf'))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    masked_weights = torch.softmax(masked_scores, dim = -1)
    return torch.nan_to_num(masked_weights, nan=0.0)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    return attention_weights@value

# Step 22 - scaled_dot_product_attention
import torch
import math
def scaled_dot_product_attention(query, key, value, mask=None):
    attention_scores = query@key.transpose(-2, -1)
    d_k  = query.shape[-1]
    scaled_attention_scores = attention_scores/math.sqrt(d_k)
    if mask is not None :
        masked_attention_scores = scaled_attention_scores.masked_fill(mask == False, float('-inf'))
    else :
        masked_attention_scores = scaled_attention_scores

    attention_weights = torch.softmax(masked_attention_scores, dim=-1)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    context =  attention_weights@value
    return context, attention_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    new_shape = (tensor.shape[0], tensor.shape[1], num_heads, tensor.shape[-1]//num_heads)
    return tensor.reshape(new_shape)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    return split_tensor.transpose(1,2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    multi_head_tensor = multi_head_tensor.transpose(1,2)
    new_shape = (multi_head_tensor.shape[0], multi_head_tensor.shape[1], multi_head_tensor.shape[2]*multi_head_tensor.shape[3])
    return multi_head_tensor.reshape(new_shape)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    if bias is not None:
        return x@weight.transpose(-2,-1) + bias 
    else :
        return x@weight.transpose(-2,-1)

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    Q = x@w_q.transpose(-2, -1) + (b_q if b_q is not None else 0) 
    K = x@w_k.transpose(-2, -1) + (b_k if b_k is not None else 0) 
    V = x@w_v.transpose(-2, -1) + (b_v if b_v is not None else 0)
    return Q, K, V

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    q_h = q.reshape(q.shape[0], q.shape[1], num_heads, q.shape[-1]//num_heads)
    q_h = q_h.transpose(1, 2)
    k_h = k.reshape(k.shape[0], k.shape[1], num_heads, k.shape[-1]//num_heads)
    k_h = k_h.transpose(1, 2)
    v_h = v.reshape(v.shape[0], v.shape[1], num_heads, v.shape[-1]//num_heads)
    v_h = v_h.transpose(1, 2)
    return q_h, k_h, v_h

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    attention_scores = q_h@k_h.transpose(-2,-1)
    d_k = torch.tensor(q_h.shape[-1])
    scaled_attention_scores = attention_scores/torch.sqrt(d_k)
    if mask is not None :
        masked_attention_scores  = scaled_attention_scores.masked_fill(mask==False, float('-inf')) 
    else :
        masked_attention_scores = scaled_attention_scores
    attention_weights = torch.softmax(masked_attention_scores, dim=-1)
    attention_weights = torch.nan_to_num(attention_weights, nan = 0.0)
    context = attention_weights@v_h
    return context, attention_weights

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o=None):
    context = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(context, w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
import traceback
import sys
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    try: 
        query = query@w_q.transpose(-2,-1)
        key = key@w_k.transpose(-2,-1)
        value = value@w_v.transpose(-2,-1)
        q_h, k_h, v_h = split_qkv_into_heads(query, key, value, num_heads)
        context, _ = multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask)
        return merge_heads_and_project_output(context, w_o)
    
    except Exception:
        traceback.print_exc()
        raise

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    z = x@w1 + b1
    return torch.relu(z)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    return hidden@w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
   x1 = apply_ffn_first_linear_and_relu(x,w1, b1)
   return apply_ffn_second_linear(x1, w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    return torch.mean(x, dim=-1,  keepdim=True), torch.var(x, dim=-1, unbiased = False,  keepdim=True)

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
   mean, var = compute_layer_norm_mean_and_variance(x)
   x_cap = (x-mean) / torch.sqrt(var + eps)
   return gamma*x_cap + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    vec = residual_input+sublayer_output
    return normalize_and_scale_with_gamma_beta(vec, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    return x*keep_mask/keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    sublayer_output = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(x, sublayer_output, gamma, beta)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    sublayer_ffn_output = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    return apply_residual_add_and_norm(x, sublayer_ffn_output, gamma, beta)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    w_q = layer_params['w_q']
    w_k = layer_params['w_k']
    w_v = layer_params['w_v']
    w_o = layer_params['w_o']
    attn_gamma = layer_params['attn_gamma']
    attn_beta = layer_params['attn_beta']
    attention_output = encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, attn_gamma, attn_beta, num_heads, src_mask)
    
    w1 = layer_params['w1']
    b1 = layer_params['b1']
    w2 = layer_params['w2']
    b2 = layer_params['b2']
    ffn_gamma = layer_params['ffn_gamma'] 
    ffn_beta = layer_params['ffn_beta']

    return encoder_layer_feed_forward_sublayer(attention_output, w1, b1, w2, b2, ffn_gamma, ffn_beta)

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    enc_output = x
    for i in range(len(encoder_layer_params_list)) :
        enc_output = assemble_encoder_layer(enc_output, encoder_layer_params_list[i], num_heads, src_mask)
    return enc_output

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    sublayer_output = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)
    return apply_residual_add_and_norm(y, sublayer_output, gamma, beta)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    if src_mask is not None and src_mask.dim != 4 :
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
    sublayer_output = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(y, sublayer_output, gamma, beta)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    sub_layer_output = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    return apply_residual_add_and_norm(y, sub_layer_output, gamma, beta)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    # print(layer_params)
    # return torch.tensor(5)
    w_q_self = layer_params['w_q_self']
    w_k_self = layer_params['w_k_self']
    w_v_self = layer_params['w_v_self']
    w_o_self = layer_params['w_o_self']
    self_gamma = layer_params['self_gamma']
    self_beta = layer_params['self_beta']
    output =  decoder_layer_masked_self_attention_sublayer(y, w_q_self, w_k_self, w_v_self, w_o_self,self_gamma, self_beta, num_heads, tgt_mask)
    
    w_q_cross = layer_params['w_q_cross']
    w_k_cross = layer_params['w_k_cross']
    w_v_cross = layer_params['w_v_cross']
    w_o_cross = layer_params['w_o_cross']
    cross_gamma = layer_params['cross_gamma']
    cross_beta = layer_params['cross_beta']
    output =  decoder_layer_cross_attention_sublayer(output, encoder_output, w_q_cross, w_k_cross, w_v_cross, w_o_cross, cross_gamma, cross_beta, num_heads, src_mask)
    
    w1 = layer_params['w1']
    b1 = layer_params['b1']
    w2 = layer_params['w2']
    b2 = layer_params['b2']
    ffn_gamma = layer_params['ffn_gamma'] 
    ffn_beta = layer_params['ffn_beta']
    return  decoder_layer_feed_forward_sublayer(output, w1, b1, w2, b2, ffn_gamma, ffn_beta)

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    output = y
    for i in range(len(decoder_layer_params_list)) :
        output = assemble_decoder_layer(output, encoder_output, decoder_layer_params_list[i], num_heads, src_mask, tgt_mask)
    return output

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings (not yet solved)
# TODO: implement

# Step 50 - apply_log_softmax_over_vocab (not yet solved)
# TODO: implement

# Step 51 - run_transformer_forward (not yet solved)
# TODO: implement

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

