import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ArcFaceHead(nn.Module):
    """
    ArcFace head for metric learning.
    
    Implements Additive Angular Margin Loss for fingerprint recognition.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_identities: int,
        margin: float = 0.5,
        scale: float = 30.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_identities = num_identities
        self.margin = margin
        self.scale = scale
        
        # Weight matrix for each identity
        self.weight = nn.Parameter(torch.Tensor(num_identities, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute ArcFace logits.
        """
        # Normalize features and weights
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(features_norm, weight_norm)
        
        # Convert to angles
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        
        if label is not None and self.training:
            # Add margin to the target class
            one_hot = F.one_hot(label, self.num_identities).float()
            margin_one_hot = one_hot * self.margin
            
            # Apply margin in angular space
            theta_with_margin = theta + margin_one_hot
            cosine_with_margin = torch.cos(theta_with_margin)
            
            # Scale and return
            output = cosine_with_margin * self.scale
        else:
            output = cosine * self.scale
        
        return output


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss for metric learning.
    
    Combines ArcFace head with CrossEntropyLoss.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_identities: int,
        margin: float = 0.5,
        scale: float = 30.0,
    ):
        super().__init__()
        self.arcface_head = ArcFaceHead(embedding_dim, num_identities, margin, scale)
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace loss.
        """
        logits = self.arcface_head(features, labels)
        loss = self.cross_entropy(logits, labels)
        return loss
