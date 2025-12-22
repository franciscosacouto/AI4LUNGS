from torch.utils.data import Dataset
import torch
import numpy as np
# Import for image processing
from PIL import Image
from torchvision import transforms 
import io
import base64

# Retain utility function, although it won't be used inside __getitem__ anymore
def array_to_base64(npy_img):
    pil_img = Image.fromarray((npy_img * 255).astype(np.uint8)) 
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class SurvivalDataset(Dataset):
    def __init__(self, df):
        """
        df must contain:
            - 'file_path' : path to .npy image
            - '5y' : event indicator
        """
        self.df = df.reset_index()  # keep pid but use sequential indices

        # Extract outcomes
        self.event = torch.tensor(self.df['5y'].values, dtype=torch.float32)

        # --- CRITICAL: Define the DINOv2/Vision Transformer preprocessing pipeline ---
        # Assuming the .npy image is a 2D or 3D single-channel (grayscale/mask) array
        # It MUST be converted to a 3-channel (RGB) tensor for DINOv2
        self.transform = transforms.Compose([
            # 1. Convert to PIL Image (needed for standard transforms like Resize/Crop)
            # This step is handled implicitly inside __getitem__ by converting the 
            # NumPy array to a PIL image first.
            
            # 2. Resize and Center Crop to the expected input size (e.g., 224x224)
            transforms.Resize(256),              
            transforms.CenterCrop(224),          
            
            # 3. Convert PIL Image to PyTorch Tensor (moves channel dimension to front)
            transforms.ToTensor(),               
            
            # 4. Standard ImageNet/DINOv2 normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        npy_path = self.df.loc[idx, "file_path"]

        # 1. Load the NumPy array
        npy_img = np.load(npy_path)
        
        if not isinstance(npy_img, np.ndarray):
            raise ValueError(f"Loaded object is not a numpy array: {npy_path}")

        # --- CRITICAL CHANGES START HERE ---
        
        # 2. Normalize and convert the array to uint8 (0-255) for PIL
        # Assuming your .npy array is float (0-1) or int (0-max_val)
        # Ensure it's in the 0-255 range and uint8 type for PIL
       # ...
        img_min = npy_img.min()
        img_max = npy_img.max()
        
        if img_max == img_min:
            # If the image is constant, create an array of zeros (or black image)
            npy_img_norm = np.zeros_like(npy_img, dtype=np.uint8)
        else:
            # Perform Min-Max scaling to 0-255 range and convert to uint8
            npy_img_norm = (
                (npy_img - img_min) / (img_max - img_min) * 255
            ).astype(np.uint8)
        
        # Now use npy_img_norm for Image.fromarray
        pil_img = Image.fromarray(npy_img_norm, mode='L') 
# ...
        
        # 4. Convert Grayscale (L) to 3-channel (RGB) image, as DINOv2 expects RGB input
        pil_img = pil_img.convert('RGB')

        # 5. Apply the standard transformations
        img_tensor = self.transform(pil_img)

        # Return the processed tensor and the event indicator
        return img_tensor, self.event[idx]