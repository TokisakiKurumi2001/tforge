import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from TForge import TForgeConfig
from typing import Union, Tuple

# NOTE: TForgeMultiHeadAttention
class TForgeMultiHeadAttention(nn.Module):
    """ Multi-head attention runs multiple attention calculations in parallel.
    """
    def __init__(self, num_heads: int, dim_embed: int, drop_prob: float) -> None:
        super(TForgeMultiHeadAttention, self).__init__()
        assert dim_embed % num_heads == 0

        # num_head x dim_head = dim_embed
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.dim_head = dim_embed // num_heads

        # Linear operations and dropout
        self.query  = nn.Linear(dim_embed, dim_embed)
        self.key    = nn.Linear(dim_embed, dim_embed)
        self.value  = nn.Linear(dim_embed, dim_embed)
        self.output = nn.Linear(dim_embed, dim_embed)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor, y: Tensor, mask: Tensor=None) -> Tensor:
        # linear transformation in one shot per query, key, value
        query = self.query(x)
        key   = self.key  (y)
        value = self.value(y)

        # Note: max here is within a batch and it's either for target or source batch
        # (batch_size, max_sequence_length, dim_embed) =>
        # (batch_size, max_sequence_length, num_heads, dim_head) =>
        # (batch_size, num_heads, max_sequence_length, dim_head)
        batch_size = x.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        key   = key  .view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        
        if mask is not None:
            # Mask needs to have an extra dimension to be broadcastable across multiple heads
            # - TForgeEncoder self-attention: (batch_size, 1,                          1, max_source_sequence_length)
            # - TForgeDecoder self-attention: (batch_size, 1, max_target_sequence_length, max_target_sequence_length)
            mask = mask.unsqueeze(1)

        # Applies the attention function on all heads in parallel
        attn = self.attention(query, key, value, mask)

        # Restores the original shapes:
        # (batch_size, num_heads, max_sequence_length, dim_head) =>
        # (batch_size, max_sequence_length, num_heads, dim_head) =>
        # (batch_size, max_sequence_length, dim_embed)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_embed)
        
        # Finally, applies one more linear operation and dropout
        out = self.output(attn)
        return out

    def attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None) -> Tensor:
        sqrt_dim_head = query.shape[-1]**0.5 # sqrt(dim_head)

        # Scaled Dot-Product by matrix operation: Q K^T / sqrt(dim_head)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / sqrt_dim_head
        
        if mask is not None:
            # Sets large negative value to masked token positions - softmax will give effectively zero probability to them.
            scores = scores.masked_fill(mask==0, -1e9)
        
        # Attention weighted value
        weight = F.softmax(scores, dim=-1)
        weight_prob = self.dropout(weight)
        return torch.matmul(weight_prob, value)

# NOTE: FeedForward
class TForgePositionwiseFeedForward(nn.Module):
    def __init__(self, dim_embed: int, dim_pffn: int, drop_prob: float) -> None:
        super(TForgePositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_embed, dim_pffn)
        self.activation_fn = nn.GELU()
        self.dropout1 = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(dim_pffn, dim_embed)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout1(self.activation_fn(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x

# NOTE: Positional encoding
class TForgeLearnedPositionalEncoding(nn.Module):
    def __init__(self, max_positions: int, dim_embed: int, drop_prob: float) -> None:
        super(TForgeLearnedPositionalEncoding, self).__init__()
        self.offset = 2
        self.pos_embedding = nn.Embedding(max_positions + self.offset, dim_embed)
        self.dropout = nn.Dropout(p=drop_prob)
        positions = torch.arange(0, 0 + max_positions, dtype=torch.long)
        self.register_buffer('positions', positions)
        
    def forward(self, input_ids: Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        positions = self.positions[:seq_len].expand(bsz, -1)
        return self.dropout(self.pos_embedding(positions + self.offset))

# NOTE: TForgeEmbedding
class TForgeEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim_embed: int, pad_token_id: int) -> None:
        super(TForgeEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, dim_embed, pad_token_id)
        self.sqrt_dim_embed = math.sqrt(dim_embed)

    def forward(self, x: Tensor) -> Tensor:  
        x = self.embedding(x.long()) # (batch_size, max_sequence_length, dim_embed)    
        x = x * self.sqrt_dim_embed  # Scaling
        return x

# NOTE: TForgeEncoder layer
class TForgeEncoderLayer(nn.Module):
    def __init__(self,
                 num_heads: int,
                 dim_embed: int,
                 dim_pwff:  int,
                 drop_prob: float) -> None:
        super(TForgeEncoderLayer, self).__init__()

        # Self-attention
        self.self_atten = TForgeMultiHeadAttention(num_heads, dim_embed, drop_prob)
        self.dropout_attn = nn.Dropout(drop_prob)
        self.layer_norm1 = nn.LayerNorm(dim_embed)

        # Point-wise feed-forward
        self.feed_forward = TForgePositionwiseFeedForward(dim_embed, dim_pwff, drop_prob)
        self.layer_norm2 = nn.LayerNorm(dim_embed)

    def forward(self, x: Tensor, x_mask: Tensor) -> Tensor:
        x = self.sub_layer1(x, x_mask)
        x = self.sub_layer2(x)
        return x

    def sub_layer1(self, x: Tensor, x_mask: Tensor) -> Tensor:
        x = x + self.dropout_attn(self.self_atten(x, x, x_mask))
        return self.layer_norm1(x)
    
    def sub_layer2(self, x: Tensor) -> Tensor:
        x = x + self.feed_forward(x)
        return self.layer_norm2(x)

# NOTE: TForgeEncoder
class TForgeEncoder(nn.Module):
    def __init__(self,
                 input_vocab_size: int,
                 max_positions: int,
                 num_layers: int,
                 num_heads:  int,
                 dim_embed:  int,
                 dim_pffn:   int,
                 drop_prob:  float,
                 pad_token_id: int) -> None:
        super(TForgeEncoder, self).__init__()
        self.embed_tokens = TForgeEmbedding(input_vocab_size, dim_embed, pad_token_id)
        self.embed_positions = TForgeLearnedPositionalEncoding(max_positions, dim_embed, drop_prob)

        self.layers = nn.ModuleList(
            [TForgeEncoderLayer(num_heads, dim_embed, dim_pffn, drop_prob) for _ in range(num_layers)]
        )
        self.layer_norm_embed = nn.LayerNorm(dim_embed)
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x: Tensor, x_mask: Tensor, pos_enc: bool):
        # x.shape=(bsz, seq_len)
        x = self.embed_tokens(x)
        if pos_enc:
            pos = self.embed_positions(x)
            x = x + pos
        x = self.dropout(self.layer_norm_embed(x))
        for layer in self.layers:
            x = layer(x, x_mask)
        return x

# NOTE: TForgeDecoder layer
class TForgeDecoderLayer(nn.Module):
    def __init__(self,
                 num_heads: int,
                 dim_embed: int,
                 dim_pwff:  int,
                 drop_prob: float) -> None:
        super(TForgeDecoderLayer, self).__init__()

        # Self-attention
        self.self_atten = TForgeMultiHeadAttention(num_heads, dim_embed, drop_prob)
        self.layer_norm1 = nn.LayerNorm(dim_embed)
        self.dropout1 = nn.Dropout(drop_prob)

        # Target-source
        self.target_source_attn = TForgeMultiHeadAttention(num_heads, dim_embed, drop_prob)
        self.layer_norm2 = nn.LayerNorm(dim_embed)
        self.dropout2 = nn.Dropout(drop_prob)

        # Position-wise
        self.feed_forward = TForgePositionwiseFeedForward(dim_embed, dim_pwff, drop_prob)
        self.layer_norm3 = nn.LayerNorm(dim_embed)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, y, y_mask, x, x_mask) -> Tensor:
        y = self.sub_layer1(y, y_mask)
        y = self.sub_layer2(y, x, x_mask)
        y = self.sub_layer3(y)
        return y

    def sub_layer1(self, y: Tensor, y_mask: Tensor) -> Tensor:
        a = self.self_atten(y, y, y_mask)
        y = y + self.dropout1(self.self_atten(y, y, y_mask))
        return self.layer_norm1(y)

    def sub_layer2(self, y: Tensor, x: Tensor, x_mask: Tensor) -> Tensor:
        y = y + self.dropout2(self.target_source_attn(y, x, x_mask))
        return self.layer_norm2(y)

    def sub_layer3(self, y: Tensor) -> Tensor:
        y = y + self.feed_forward(y)
        return self.layer_norm3(y)

# NOTE: TForgeDecoder
class TForgeDecoder(nn.Module):
    def __init__(self,
                 output_vocab_size: int,
                 max_positions: int,
                 num_layers: int,
                 num_heads:  int,
                 dim_embed:  int,
                 dim_pffn:   int,
                 drop_prob:  float,
                 pad_token_id: int) -> None:
        super(TForgeDecoder, self).__init__()
        self.embed_tokens = TForgeEmbedding(output_vocab_size, dim_embed, pad_token_id)
        self.embed_positions = TForgeLearnedPositionalEncoding(max_positions, dim_embed, drop_prob)

        self.layers = nn.ModuleList(
            [TForgeDecoderLayer(num_heads, dim_embed, dim_pffn, drop_prob) for _ in range(num_layers)]
        )
        self.layer_norm_embed = nn.LayerNorm(dim_embed)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor, x_mask: Tensor, y: Tensor, y_mask: Tensor) -> Tensor:
        # x.shape=(bsz, seq_len)
        y = self.embed_tokens(y)
        pos = self.embed_positions(y)
        y = self.dropout(self.layer_norm_embed(y + pos))
        for layer in self.layers:
            y = layer(y, y_mask, x, x_mask)
        return y

# NOTE: TForgeCopyMechanism
class TForgeSoftAlign(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        soft_align_dim: int,
        drop_prob: float):
        super(TForgeSoftAlign, self).__init__()
        self.align = nn.Linear(embed_dim, soft_align_dim)
        self.dropout1 = nn.Dropout(drop_prob)
        self.activation_fn = nn.GELU()
        self.dropout2 = nn.Dropout(drop_prob)
        self.feed_forward = nn.Linear(soft_align_dim, soft_align_dim)
        
    def forward(self, enc_hidden: Tensor, dec_hidden: Tensor):
        align_enc = self.align_hidden(enc_hidden) # (b, sk, soft_dim)
        align_dec = self.align_hidden(dec_hidden) # (b, st, soft_dim)
        align_matrix = F.softmax(torch.bmm(align_dec, align_enc.permute(0, 2, 1)), dim=-1) # (b, st, sk)
        return align_matrix

    def align_hidden(self, input: Tensor) -> Tensor:
        out = self.dropout1(self.activation_fn(self.align(input)))
        out = self.dropout2(self.activation_fn(self.feed_forward(out)))
        return out

class TForgeCopyLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        drop_prob: float):
        super(TForgeCopyLayer, self).__init__()
        self.cross_attn = TForgeMultiHeadAttention(num_heads, embed_dim, drop_prob)
        self.p_gen = nn.Linear(embed_dim, 1)
    
    def forward(self, enc_hidden: Tensor, dec_hidden: Tensor, enc_mask: Tensor):
        att = self.cross_attn(dec_hidden, enc_hidden, enc_mask)
        p_gen = torch.sigmoid(self.p_gen(att)) # p_gen = (bz, seq, 1)
        return p_gen

# NOTE: Transformer model
class TForgePretrainedModel(PreTrainedModel):
    config_class = TForgeConfig
    base_model_prefix = "Transformer_Forge"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Embedding]):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.TForgeEmbedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (TForgeDecoder, TForgeEncoder)):
            module.gradient_checkpointing = value

# Main model used for training
class TForgeModel(TForgePretrainedModel):
    def __init__(self, config: TForgeConfig):
        super(TForgeModel, self).__init__(config)

        self.encoder = TForgeEncoder(
            self.config.input_vocab_size,
            self.config.max_positions,
            self.config.num_layers,
            self.config.num_heads,
            self.config.embed_dim,
            self.config.hidden_dim,
            self.config.dropout_prob,
            self.config.pad_token_id)
        self.decoder_tgt = TForgeDecoder(
            self.config.output_vocab_size,
            self.config.max_positions,
            self.config.num_layers,
            self.config.num_heads,
            self.config.embed_dim,
            self.config.hidden_dim,
            self.config.dropout_prob,
            self.config.pad_token_id)
        self.decoder_tsl = TForgeDecoder(
            self.config.translate_vocab_size,
            self.config.max_positions,
            self.config.num_layers,
            self.config.num_heads,
            self.config.embed_dim,
            self.config.hidden_dim,
            self.config.dropout_prob,
            self.config.pad_token_id)
        self.copy_layer = TForgeCopyLayer(
            self.config.num_heads,
            self.config.embed_dim,
            self.config.dropout_prob)
        self.soft_align = TForgeSoftAlign(
            self.config.embed_dim,
            self.config.soft_align_dim,
            self.config.dropout_prob)
        self.projection_tgt = nn.Linear(self.config.embed_dim, self.config.output_vocab_size)
        self.projection_tsl = nn.Linear(self.config.embed_dim, self.config.translate_vocab_size)

        # Initialize parameters
        if not self.config.load_pretrained:
            for param in self.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            self.config.load_pretrained = True

    def forward(self, input) -> Tuple[Tensor, Tensor]:
        encoder_inp, encoder_inp_mask = input['encoder_input_ids'], input['encoder_input_mask']
        encoder_kw, encoder_kw_mask = input['encoder_keyword_ids'], input['encoder_keyword_mask']
        decoder_tgt, decoder_tgt_mask = input['decoder_tgt_ids'], input['decoder_tgt_mask']
        decoder_tsl, decoder_tsl_mask = input['decoder_tsl_ids'], input['decoder_tsl_mask']
        hidden_input = self.encoder(encoder_inp, encoder_inp_mask, pos_enc=True)
        hidden_kw = self.encoder(encoder_kw, encoder_kw_mask, pos_enc=False)
        hidden_tgt = self.decoder_tgt(hidden_input, encoder_inp_mask, decoder_tgt, decoder_tgt_mask)
        hidden_tsl = self.decoder_tsl(hidden_input, encoder_inp_mask, decoder_tsl, decoder_tsl_mask)
        pre_out_tgt = self.projection_tgt(hidden_tgt)
        out_tsl = self.projection_tsl(hidden_tsl)

        copy_dist = self.make_copy_dist(encoder_kw)
        tgt_src_align = self.soft_align(hidden_kw, hidden_tgt)
        copy_dist_align = torch.bmm(tgt_src_align, copy_dist)
        p_gen = self.copy_layer(hidden_kw, hidden_tgt, encoder_kw_mask)
        pre_out_tgt = F.softmax(pre_out_tgt, dim=-1)
        out_tgt = (1 - p_gen) * copy_dist_align + p_gen * pre_out_tgt
        out_tgt = torch.log(out_tgt)
        out_tsl = nn.LogSoftmax(dim=-1)(out_tsl)
        return out_tgt, out_tsl

    def make_copy_dist(self, input_ids: Tensor) -> Tensor:
        one_hot_dist = F.one_hot(input_ids, self.config.output_vocab_size)
        size = one_hot_dist.shape
        noise = torch.normal(0, 1, size=size).to(device=input_ids.device)
        return F.softmax(one_hot_dist + noise, dim=-1)

# Wrapper for text generation
class TForgeForGeneration:
    def __init__(self, model: TForgeModel, alpha: float=0.6):
        self.model = model
        self.alpha = alpha

    def __sequence_length_penalty(self, length: int, alpha: float) -> float:
            """ Sequence length penalty for beam search.
            
            Source: Google's Neural Machine Translation System (https://arxiv.org/abs/1609.08144)
            """
            return ((5 + length) / (5 + 1)) ** alpha

    def generate(self, input_ids: Tensor, num_beam: int=4, max_output_length: int=20, vocab_size: int=10000) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            encoder_output = self.model.encoder(input_ids)

            return self.__decode(encoder_output, max_output_length, vocab_size, num_beam)

    def __decode(
        self, encoder_output: Tensor, max_output_length: int, vocab_size: int, 
        num_beam: int, bos_token_id: int=2, eos_token_id: int=3
    ):
        # Start with <bos>
        decoder_input = torch.Tensor([[bos_token_id]]).long()
        scores = torch.Tensor([0.])
        for i in range(max_output_length):
            # TForgeEncoder output expansion from the second time step to the beam size
            if i==1:
                encoder_output = encoder_output.expand(num_beam, *encoder_output.shape[1:])

            # TForgeDecoder prediction
            logits = self.model.decoder(encoder_output, decoder_input)
            logits = logits[:, -1] # Last sequence step: [beam_size, sequence_length, vocab_size] => [beam_size, vocab_size]

            # Softmax
            log_probs = torch.log_softmax(logits, dim=1)
            log_probs = log_probs / self.__sequence_length_penalty(i+1, self.alpha)

            # Update score where EOS has not been reched
            log_probs[decoder_input[:, -1]==eos_token_id, :] = 0
            scores = scores.unsqueeze(1) + log_probs # scores [beam_size, 1], log_probs [beam_size, vocab_size]

            # Flatten scores from [beams, vocab_size] to [beams * vocab_size] to get top k, and reconstruct beam indices and token indices
            scores, indices = torch.topk(scores.reshape(-1), num_beam)
            beam_indices  = torch.divide   (indices, vocab_size, rounding_mode='floor') # indices // vocab_size
            token_indices = torch.remainder(indices, vocab_size)                        # indices %  vocab_size

            # Build the next decoder input
            next_decoder_input = []
            for beam_index, token_index in zip(beam_indices, token_indices):
                prev_decoder_input = decoder_input[beam_index]
                if prev_decoder_input[-1]==eos_token_id:
                    token_index = eos_token_id # once EOS, always EOS
                token_index = torch.LongTensor([token_index])
                next_decoder_input.append(torch.cat([prev_decoder_input, token_index]))
            decoder_input = torch.vstack(next_decoder_input)

            # If all beams are finished, exit
            if (decoder_input[:, -1]==eos_token_id).sum() == num_beam:
                break

        # convert the top scored sequence to a list of text tokens
        decoder_output, _ = max(zip(decoder_input, scores), key=lambda x: x[1])
        decoder_output = decoder_output[1:] # remove SOS
        return decoder_input