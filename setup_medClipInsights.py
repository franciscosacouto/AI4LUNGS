import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Optional: on Windows, you can force PyTorch to use OpenBLAS instead of MKL
# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image
import nrrd
import numpy as np
import sys
import torch
import torch
import base64
import io


scan, scan_header = nrrd.read("100012_1/100012_scan.nrrd")
mask, mask_header = nrrd.read("100012_1/100012_1_mask_corrected_normalized.nrrd")

# Apply lung mask
lung_only = scan * mask

# Normalize to [0, 1]
lung_only = (lung_only - np.min(lung_only)) / (np.max(lung_only) - np.min(lung_only))
print("Lung only volume shape:", lung_only.shape)

from PIL import Image

def get_slices(volume, every_n=1):
    slices = []
    # Iterate over the depth axis (axis=3)
    for i in range(0, volume.shape[2], every_n):
        slice_img = (volume[:, :, i] * 255).astype(np.uint8)
        pil_img = Image.fromarray(slice_img).convert("RGB")
        slices.append(pil_img)
    return slices

slices = get_slices(lung_only)
print("Prepared", len(slices), "slices.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


sys.path.insert(1, 'MedImageInsights')

from medimageinsightmodel import MedImageInsight

classifier = MedImageInsight(
    model_dir="MedImageInsights/2024.09.27",

    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)



classifier.load_model()

#get embeddings for each slice


def pil_to_base64(pil_img):
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_embedding(pil_img, classifier):
    img_base64 = pil_to_base64(pil_img)
    images = [img_base64]

    with torch.no_grad():
        emb_dict = classifier.encode(images=images)  # returns dict
        emb = emb_dict["image_embeddings"]               # extract tensor
    
    emb_tensor = torch.tensor(emb) if isinstance(emb, np.ndarray) else emb
    return emb_tensor.squeeze()


all_embeddings = [get_embedding(img, classifier) for img in slices]
all_embeddings_tensor = torch.stack(all_embeddings)
print("All embeddings shape:", all_embeddings_tensor.shape)