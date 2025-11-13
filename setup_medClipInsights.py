import os
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

import pandas as pd



def search_files(rootdir,df):
    for  file in os.walk(rootdir):
        print(os.path.join(file))
        new_row ={ "PID": file.split('_')[0], "file_path": os.path.join( file)}
        #attach file name to df
        df.append(new_row, ignore_index=True)

    return df



def get_slices_3d(volume, every_n=1):
    slices = []
    # Iterate over the depth axis (axis=3)
    for i in range(0, volume.shape[2], every_n):
        slice_img = (volume[:, :, i] * 255).astype(np.uint8)
        pil_img = Image.fromarray(slice_img).convert("RGB")
        slices.append(pil_img)
    return slices

sys.path.insert(1, '/nas-ctm01/homes/fmferreira/MedImageInsight')
from medimageinsightmodel import MedImageInsight
classifier = MedImageInsight(
    model_dir="MedImageInsights/2024.09.27",

    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)

#get embeddings for each slice

def pil_to_base64(pil_img):
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_embedding(pil_img_path, classifier):
    pil_img = Image.open(pil_img_path).convert("RGB")
    img_base64 = pil_to_base64(pil_img)
    images = [img_base64]

    with torch.no_grad():
        emb_dict = classifier.encode(images=images)  # returns dict
        emb = emb_dict["image_embeddings"]               # extract tensor
    
    emb_tensor = torch.tensor(emb) if isinstance(emb, np.ndarray) else emb
    return emb_tensor.squeeze()


def get_embeddings_from_dataframe(df, classifier):
    embeddings = {}
    for row in df.iterrows():
        pid = row['PID']
        file_path = row['file_path']
        emb = get_embedding(file_path, classifier)
        embeddings[pid] = emb
    return embeddings

#main code
def main():

    df= pd.dataframe()
    classifier.load_model()

    rootdir_lung = '/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/preprocessed_data/protocol_5/2d/lung'
    rootdir_ws = '/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/preprocessed_data/protocol_5/2d/ws'
    rootdir_masked = '/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/preprocessed_data/protocol_5/2d/masked'

    df = search_files( rootdir_lung, df)  

    embeddings = get_embeddings_from_dataframe(df, classifier)
    print(embeddings)


    
if __name__ == "__main__":
    main()



