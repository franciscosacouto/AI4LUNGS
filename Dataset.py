from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import io
import base64

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

        self.df = df.reset_index()   # keep pid but use sequential indices

        # Extract outcomes
        self.event = torch.tensor(self.df['5y'].values, dtype=torch.float32)

        # Optional: detect tabular columns besides file_path, 5y, fup_days
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        npy_path = self.df.loc[idx, "file_path"]

        npy_img = np.load(npy_path)
        

        if not isinstance(npy_img, np.ndarray):
            raise ValueError(f"Loaded object is not a numpy array: {npy_path}")

        img_b64 = array_to_base64(npy_img)

        return img_b64, self.event[idx]