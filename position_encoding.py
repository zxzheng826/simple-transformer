import torch
from torch import nn 

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list
        h, w = tensor_list.shape[-2:]
        i = torch.arange(w, device=x.device) #[0,...,w]
        j = torch.arange(h, device=x.device) #[0,...,h]
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        return pos

def build_position_encoding(hidden_dim):
    N_steps = hidden_dim // 2
    position_embedding = PositionEmbeddingLearned(N_steps)
    return position_embedding

