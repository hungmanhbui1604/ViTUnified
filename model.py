import timm
import torch
import torch.nn as nn


class PADHead(nn.Module):
    def __init__(self, dim: int, num_classes: int = 2, dropout: float = 0.0):
        super().__init__()

        hidden = dim // 2
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = x[:, 0].contiguous()  # (B, D)
        return self.head(cls_token)


class ViTUnified(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        num_classes: int = 2,
        pad_dropout: float = 0.0,
    ):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.backbone.reset_classifier(0)

        self.embed_dim = self.backbone.embed_dim
        self.num_layers = len(self.backbone.blocks)

        self.pad_heads = nn.ModuleList(
            [
                PADHead(self.embed_dim, num_classes, pad_dropout)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        # Patch embedding
        x = self.backbone.patch_embed(x)

        # Patch embedding
        cls_token = self.backbone.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_token, x), dim=1).contiguous()

        # Positional embedding
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        # Transformer blocks
        x = self.backbone.norm_pre(x)

        pad_outputs = []
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            logits = self.pad_heads[i](x)
            pad_outputs.append(logits)

        # Final embedding
        x = self.backbone.norm(x)
        final_embedding = x[:, 0].contiguous()

        return final_embedding, pad_outputs

    def feature_extraction(self, x: torch.Tensor) -> torch.Tensor:
        final_embedding, _ = self.forward(x)
        return final_embedding
    
    def _freeze_pad_heads(self):
        for param in self.pad_heads.parameters():
            param.requires_grad = False
        
        self.pad_heads.eval()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone.eval()
