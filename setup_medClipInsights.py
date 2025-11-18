import os
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image
import numpy as np
import sys
import base64
import io
import hydra
from hydra import initialize, compose
import pandas as pd



def search_files(rootdir, df):
    for dirpath, _, filenames in os.walk(rootdir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            pid = filename.split('_')[0]  # get PID from filename
            new_row = {"PID": pid, "file_path": full_path}
            df.loc[len(df)] = new_row
            print(f"Found file: {full_path}")
    return df



# def get_slices_3d(volume, every_n=1):
#     slices = []
#     # Iterate over the depth axis (axis=3)
#     for i in range(0, volume.shape[2], every_n):
#         slice_img = (volume[:, :, i] * 255).astype(np.uint8)
#         pil_img = Image.fromarray(slice_img).convert("RGB")
#         slices.append(pil_img)
#     return slices

sys.path.insert(1, '/nas-ctm01/homes/fmferreira/MedImageInsights')
from medimageinsightmodel import MedImageInsight
classifier = MedImageInsight(
    model_dir="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27",

    vision_model_name="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27/vision_model/medimageinsigt-v1.0.0.pt",
    language_model_name="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27/language_model/language_model.pth"
)

#get embeddings for each slice

def pil_to_base64(pil_img):
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_embedding(npy_img_path, classifier):
    npy_img = np.load(npy_img_path)
    # Assuming npy_img is a 2D array (single slice), convert to PIL image
    pil_img = Image.fromarray((npy_img * 255).astype(np.uint8)) 
    img_base64 = pil_to_base64(pil_img)
    images = [img_base64]

    with torch.no_grad():
        emb_dict = classifier.encode(images=images)  # returns dict
        emb = emb_dict["image_embeddings"]               # extract tensor
    
    emb_tensor = torch.tensor(emb) if isinstance(emb, np.ndarray) else emb
    return emb_tensor.squeeze()


def get_embeddings_from_dataframe(df, classifier):
    embeddings = {}
    for _, row in df.iterrows():
        pid = row['PID']
        file_path = row['file_path']
        emb = get_embedding(file_path, classifier)
        embeddings[pid] = emb
    return embeddings

#main code
@hydra.main(version_base=None,
             config_path=".", 
             config_name="config")
def main(config):

    df= pd.DataFrame( columns=[config.data.columns.pid, 
                               config.data.columns.file_path])
    classifier.load_model()

    rootdir_lung = config.directories.rootdir_lung
    rootdir_ws = config.directories.rootdir_ws
    rootdir_masked = config.directories.rootdir_masked
    print("Searching lung files...")
    df = search_files( rootdir_lung, df)  
    print("starting embeddings extraction...")
    embeddings = get_embeddings_from_dataframe(df, classifier)


    #save embeddings
    output_file = 'nlst_lung_ct_embeddings_protocol_5_2d_lung.pt'
    torch.save(embeddings, output_file)   

    #save as csv pid and embeddings
    emb_list = []
    for pid, emb in embeddings.items():
        emb_list.append({'PID': pid, 'embedding': emb.numpy()}) 
    emb_df = pd.DataFrame(emb_list)
    emb_df.to_csv('nlst_lung_ct_embeddings_protocol_5_2d_lung.csv', index=False)
    print(f"Embeddings saved to {output_file} and nlst_lung_ct_embeddings_protocol_5_2d_lung.csv")
    
if __name__ == "__main__":
    main()



