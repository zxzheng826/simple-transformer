import torch
from torch import nn 
from torch import functional as F
import backbone
import simple_transformer
import position_encoding


class simple_transNet(nn.Module):

    def __init__(self, num_classes,
                 hidden_dim, nheads,
                 num_encoder_layers, num_decoder_layers,
                 device) -> None:
        super().__init__()
        self.backbone = backbone.simple_backbone(3, 128) 
        self.pos = position_encoding.build_position_encoding(hidden_dim)
        self.input_proj = nn.Conv2d(128, hidden_dim, 1)
        
        self.query_embed = nn.Embedding(num_classes, hidden_dim)
        self.transformer = simple_transformer.Transformer(hidden_dim, nheads, num_encoder_layers,num_decoder_layers, 128, "relu", 0.1)
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        self.device = device

        
        self.class_embed = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        """
        inputs shape should be : [batch_size, 3, H, W]
        mask shape should be : [batch_size, H, W]
        """
        b,c,h,w = inputs.shape()
        mask = torch.zeros((b,h,w), dtype=torch.bool, device=self.device)
        for img, m in zip(inputs, mask):
            m[: img.shape[1], : img.shape[2]] = False
        features, m = self.backbone(inputs, mask)
        pox = []
        for x in features.items():
            pox.append(self.pos(x).to(x.dtype))

        proj_f = self.input_proj(features)
        hs = self.transformer(src=proj_f, mask=m, query_embed=self.query_embed.weight, pos_embed=pox[-1])[0]

        outputs_logit = self.class_embed(hs)
        
        return outputs_logit

def post_process( prob_logit):
    prob = F.softmax(prob_logit, -1)
    scores, labels = prob[..., :-1].max(-1)
    results = [{'scores': s, 'labels': l} for s, l, b in zip(scores, labels)]
    return results




