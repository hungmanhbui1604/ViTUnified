import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import timm


class ViTUnified(nn.Module):    
    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes_pad: int = 2,
        **kwargs
    ):
        super().__init__()
        # Load the base model without the final classification head (num_classes=0)
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            **kwargs
        )
        self.embed_dim = self.model.num_features
        self.num_blocks = len(self.model.blocks)
        
        # PAD classifier heads after each transformer layer (block)
        self.pad_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(),
                nn.Linear(self.embed_dim // 2, num_classes_pad)
            ) for _ in range(self.num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning final embedding and a list of PAD logits from each layer.
        """
        # Standard ViT forward path but intercepting each block's output
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        
        if hasattr(self.model, 'patch_drop'):
            x = self.model.patch_drop(x)
        if hasattr(self.model, 'norm_pre'):
            x = self.model.norm_pre(x)
        
        pad_outputs = []
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            # Use CLS token (index 0) for intermediate PAD classification
            cls_token = x[:, 0]
            pad_outputs.append(self.pad_heads[i](cls_token))
            
        x = self.model.norm(x)
        # Final CLS representation (features)
        final_embedding = self.model.forward_head(x, pre_logits=True)
        
        return final_embedding, pad_outputs
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only the final feature embeddings (CLS token)."""
        features = self.model.forward_features(x)
        return self.model.forward_head(features, pre_logits=True)


def vit_small_unified(pretrained: bool = True, num_classes_pad: int = 2, **kwargs) -> ViTUnified:
    """Create a small unified ViT (ViT-Small/16) with PAD heads."""
    return ViTUnified(
        model_name="vit_small_patch16_224", 
        pretrained=pretrained, 
        num_classes_pad=num_classes_pad, 
        **kwargs
    )


if __name__ == "__main__":
    from loss import ArcFaceHead
    
    try:
        model = vit_small_unified(pretrained=True)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Forward pass
        x = torch.randn(4, 3, 224, 224)
        embedding, pad_logits = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Final embedding shape: {embedding.shape}")
        print(f"Number of PAD heads: {len(pad_logits)}")
        print(f"PAD logit shape (from head 0): {pad_logits[0].shape}")
        
        # Test feature extraction
        features = model.extract_features(x)
        print(f"Extracted feature shape: {features.shape}")
        
        # Test ArcFace head
        arcface = ArcFaceHead(embedding_dim=model.embed_dim, num_identities=100)
        logits = arcface(features)
        print(f"ArcFace logits shape: {logits.shape}")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
