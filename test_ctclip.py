import torch
import sys
import os
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoTokenizer
# 1. Path Setup: Ensure this points to the folder containing the 'ct_clip' folder
# Based on your previous trace, it's /CT-CLIP/CT_CLIP
sys.path.insert(0, '/nas-ctm01/homes/fmferreira/CT-CLIP/CT_CLIP')

from ct_clip import CTCLIP
# 2. FORCE CPU - This bypasses the GTX 1080 compatibility error
device = torch.device("cpu")
print(f"Using device: {device}")

# 3. Create a Dummy Image Encoder to save RAM
class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
    def forward(self, x, *args, **kwargs): return x


# 4. Initialize
model = CTCLIP(
    image_encoder = DummyEncoder(), 
    dim_image = 294912,
    dim_text = 768, 
    dim_latent = 512,
    extra_latent_projection = True
)

# 5. Load Weights (Crucial: map_location='cpu')
weights_path = "/nas-ctm01/homes/fmferreira/CT-CLIP/CT_CLIP/ct_clip/CT-CLIP_v2.pt"
checkpoint = torch.load(weights_path, map_location=device)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
model.load_state_dict(state_dict, strict=False)

model.to(device) # Ensure model is on CPU
model.eval()

# 6. Setup Tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True, trust_remote_code=True)


# Verify it's the right class
print(f"Loaded tokenizer class: {tokenizer.__class__.__name__}")
# 7. Extract Text Embeddings
text_input = ["No evidence of acute intracranial hemorrhage or mass effect."]

with torch.no_grad():
    # Tokenize and ensure tensors are on CPU
    tokens = tokenizer(
        text_input, 
        padding="max_length", 
        truncation=True, 
        max_length= 256, 
        return_tensors="pt"
    ).to(device) 
    
    # Run through the text transformer
    # We use tokens.input_ids (not tokens['input_ids']) for better compatibility

    max_pos = model.text_transformer.abs_pos_emb.num_embeddings
    print(f"The model's max sequence length is: {max_pos}")
    enc_text = model.text_transformer(tokens.input_ids)


    
    # Your code's TextTransformer returns the full sequence. 
    # We extract the [CLS] token (index 0) as seen in your forward() source code.
    text_embeds = enc_text[:, 0, :]
    
    # Project to the 512-dim CLIP latent space
    text_latents = model.to_text_latent(text_embeds)
    
    # L2 Normalize
    text_latents = torch.nn.functional.normalize(text_latents, dim=-1)

print(f"Final Embedding Shape: {text_latents.shape}")
print("Embeddings ready for use!")