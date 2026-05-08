import torch
import torch.nn as nn
import timm
import sys
from types import SimpleNamespace
from transformers import BertModel

# Path setup
sys.path.insert(1, '/nas-ctm01/homes/fmferreira/MedImageInsights')
sys.path.insert(0, '/nas-ctm01/homes/fmferreira/CT-CLIP/CT_CLIP')
from transformer_maskgit import CTViT

# --- The Shield: Prevents the 2.7TB Allocation Error ---
class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
    def forward(self, x, *args, **kwargs): return x

def get_encoders(image_encoder_name, text_encoder_name, device):
    # Dictionary to hold the objects
    encoders = {'vision': None, 'language': None}


    if 'CTCLIP' in [image_encoder_name, text_encoder_name]:
        from ct_clip import CTCLIP as  realCTCLIP

        text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
        image_encoder = CTViT(
        dim=512, codebook_size=8192, image_size=480, patch_size=20,
        temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
        dim_head=32, heads=8
        )

        # We pass DummyEncoder here to stop CTCLIP from building its own massive transformer
        ct_model = realCTCLIP(
            image_encoder=DummyEncoder(), 
            text_encoder = text_encoder,
            dim_image=294912, dim_text=768, dim_latent=512,
            extra_latent_projection=True
        )
        
        weights_path = "/nas-ctm01/homes/fmferreira/CT-CLIP/CT_CLIP/ct_clip/CT-CLIP_v2.pt"
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # strict=False is necessary because we are 'missing' the real visual weightss
        ct_model.load_state_dict(state_dict, strict=False)
        ct_model.to(device)

        if image_encoder_name == 'CTCLIP':
            encoders['vision'] = ct_model
        if text_encoder_name == 'CTCLIP':
            encoders['language'] = ct_model

    # 1. MedImageInsights
    if 'MedImageInsights' in [image_encoder_name, text_encoder_name]:
        from medimageinsightmodel import MedImageInsight
        med_model = MedImageInsight(
            model_dir="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27",
            vision_model_name="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27/vision_model/medimageinsigt-v1.0.0.pt",
            language_model_name="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27/language_model/language_model.pth"
        )
        med_model.load_model()
        if image_encoder_name == 'MedImageInsights':
            encoders['vision'] = med_model
        if text_encoder_name == 'MedImageInsights':
            encoders['language'] = med_model

    # 2. CT-CLIP
    
    # 3. RadioDino
    if image_encoder_name == 'RadioDino':
        encoders['vision'] = timm.create_model("hf_hub:Snarcy/RadioDino-s16", pretrained=True).to(device)

    return encoders

def main():
    image_encoder_name = 'MedImageInsights'
    text_encoder_name = ''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting test on {device}...")
    encoders = get_encoders(image_encoder_name, text_encoder_name, device)

    print("\n--- FINAL VERIFICATION ---")
    if encoders['vision'] is not None:
        print(f"✅ VISION LOADED: {image_encoder_name}")
    if encoders['language'] is not None:
        print(f"✅ LANGUAGE LOADED: {text_encoder_name}")
    print("--- TEST COMPLETE ---")

if __name__ == "__main__":
    main()