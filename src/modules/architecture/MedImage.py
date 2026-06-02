import torch
import torch.nn as nn

class MedicalVisionEncoder(nn.Module):
    def __init__(self, vision_backbone):
        super().__init__()
        
        # Keep our safe unpacking logic from before
        if hasattr(vision_backbone, 'model') and hasattr(vision_backbone.model, 'image_encoder'):
            self.vision_encoder = vision_backbone.model.image_encoder
        elif hasattr(vision_backbone, 'image_encoder'):
            self.vision_encoder = vision_backbone.image_encoder
        else:
            self.vision_encoder = vision_backbone

    def forward(self, images):
        # FIX: Remove ['images'] indexing. 
        # 'images' is already the raw tensor passed from fusion.py
        v_out = self.vision_encoder(images) 
        
        # Handle pooler outputs safely
        img_emb = v_out.pooler_output if hasattr(v_out, "pooler_output") else v_out
        
        if isinstance(img_emb, (list, tuple)): 
            img_emb = img_emb[0][:, 0, :]
            
        return img_emb.view(img_emb.size(0), -1)