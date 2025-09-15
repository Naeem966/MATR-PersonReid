import torch
import torch.nn as nn
from transformers import ViTModel, BertModel

class RegionAttention(nn.Module):
    def __init__(self, dim, num_parts=3, heads_per_part=2):
        super().__init__()
        self.num_parts = num_parts
        self.attn = nn.ModuleList([nn.MultiheadAttention(dim, heads_per_part) for _ in range(num_parts)])

    def forward(self, x):
        B, C, H, W = x.shape
        region_slices = torch.chunk(x, self.num_parts, dim=2)
        region_features = []
        for i, region in enumerate(region_slices):
            region = region.flatten(2).permute(2, 0, 1)
            out, _ = self.attn[i](region, region, region)
            out = out.permute(1, 2, 0).reshape(B, C, region_slices[i].size(2), W)
            region_features.append(out)
        return torch.cat(region_features, dim=2)

class MAFormer(nn.Module):
    def __init__(self, num_classes, attr_dim=768, num_parts=3):
        super().__init__()
        self.vit = ViTModel.from_pretrained('/home/tq_naeem/Project/.venv/main/')
        self.text_encoder = BertModel.from_pretrained('/home/tq_naeem/Project/.venv/main/Sentence/')
        self.region_attn = RegionAttention(dim=768, num_parts=num_parts)
        self.cross_attn = nn.MultiheadAttention(768, 8)
        self.head = nn.Linear(768, num_classes)
        # Add this line if your attribute embeddings are 384-dimensional!
        self.attr_proj = nn.Linear(384, 768)

    def forward(self, img, attr_emb, return_features=False):
        vis_out = self.vit(img).last_hidden_state
        vis_feat = vis_out[:, 1:, :].transpose(1, 2)
        N = vis_feat.shape[2]
        H = W = int(N ** 0.5)
        vis_feat = vis_feat.reshape(-1, 768, H, W)
        region_feat = self.region_attn(vis_feat).flatten(2).mean(-1)
        # Project attr_emb if needed
        if attr_emb.shape[-1] != 768:
            attr_emb = self.attr_proj(attr_emb)
        region_feat = region_feat.unsqueeze(0)
        attr_emb = attr_emb.unsqueeze(0)
        fused, _ = self.cross_attn(region_feat, attr_emb, attr_emb)
        fused = fused.squeeze(0)
        if return_features:
            return fused
        logits = self.head(fused)
        return logits