from typing import Optional
import torch
import copy
import torch.nn.functional as F
from torch import nn, Tensor

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Transformer(nn.Module):
    def __init__(self, model_dim = 32, n_head = 5,
                 num_encoder_layers = 6, num_decoder_layers = 6, dim_feedforward = 2048,
                 activation = "relu", dropout_rate = 0.1):
        
        """
        A simple transformer for image recognition.
        transformer model dimension size: model_dim = 32
        attention head numbers: n_head = 4
        """
        super().__init__()

        self.model_dim = model_dim
        self.n_head = n_head

        encoder_layer = TransformerEncoderLayer(model_dim, n_head, d_feedforward=2048)
        encoder_norm = nn.LayerNorm(model_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(model_dim, n_head, dim_feedforward, dropout_rate, activation)
        decoder_norm = nn.LayerNorm(model_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

    
    def _reset_parameter(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        src: input [Batch_size, Channels, Hight, Width]
        mask: mask the previous output
        query_embed: embeded information from the origin input, used for decoder to gain origin information 
        pos_embed: the positional information in the input map
        """
        batch_size, channel, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)# N*C*H*W -> N*C*HW -> HW*N*C
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1 ,batch_size, 1)
        mask = mask.flatten(1)

        tgt =  torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask = mask, pos = pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                        pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1,2)
        memory = memory.permute(1, 2, 0).view(batch_size, channel, h, w)
        return hs, memory

class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, nhead, d_feedforward = 2048, dropout = 0.1, acti = "relu"):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(model_dim, d_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, model_dim)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation=acti)
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, 
                src_mask:Optional[Tensor] = None,
                src_key_padding_mask:Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        src_key_padding:if provided, specified padding elements in the key will be ignored by the attention. This is an binary mask. When the value is True, the corresponding value on the attention layer will be filled with -inf.
        src_mask:mask that prevents attention to certain positions. This is an additive mask (i.e. the values will be added to the attention layer).
        """
        
        q = k = self.with_pos_embed(src, pos) # HW * N * C?
        src2 = self.self_attention(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]#outputs of self_attension are attn_output, attn_output_weights
        src = src +self.dropout1(src2)
        x = self.linear1(src)
        x = self.activation(x)
        x = self.dropout(x)
        src2 = self.linear2(x)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_encoder_layers, encoder_norm = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_encoder_layers)
        self.num_layers = num_encoder_layers
        self.norm = encoder_norm
    
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output
        

class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, nhead, d_feedforward, dropout, acti):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(model_dim, d_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, model_dim)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(acti)
    
    def with_pos_embed(self, tensor, pos:Optional[Tensor]):
        return tensor if pos is None else pos + tensor

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt,query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        x = self.linear1(tgt)
        x = self.activation(x)
        x = self.dropout(x)
        tgt2 = self.linear2(x)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

                

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_decoder_layers, decoder_norm = None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_decoder_layers)
        self.num_layers = num_decoder_layers
        self.norm = decoder_norm
    
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None
                ):
        
        output = tgt
        for layer in self.layers:
            output = layer(tgt, memory, 
                            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask, 
                            pos=pos, query_pos=query_pos)
        if self.norm is not None:
            output = self.norm(output)

        return output.unsqueeze(0)


