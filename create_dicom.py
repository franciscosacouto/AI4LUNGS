import os
import numpy as np
import SimpleITK as sitk

# 1. Define the source path (protocol_5)
rootdir = '/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/preprocessed_data/protocol_5/2d/ws'

# 2. Automatically find the parent directory and set protocol_7
# This gets the path up to 'preprocessed_data'
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(rootdir))) 
output_dir = os.path.join(parent_dir, 'protocol_7', '2d', 'ws')

# Create the full path (including subfolders)
os.makedirs(output_dir, exist_ok=True)

print(f"Source: {rootdir}")
print(f"Destination: {output_dir}")

# 3. Process the files
for dirpath, _, filenames in os.walk(rootdir):
    for filename in filenames:
        if filename.endswith('.npy'):
            filepath = os.path.join(dirpath, filename)
            pixel_array = np.load(filepath)
            
            # Convert to DICOM format (assuming CT int16 range)
            sitk_img = sitk.GetImageFromArray(pixel_array.astype(np.int16))

            
            # Save to the new protocol_7 path
            dcm_filename = filename.replace('.npy', '.dcm')
            output_path = os.path.join(output_dir, dcm_filename)
            sitk.WriteImage(sitk_img, output_path)

print("Conversion to Protocol 7 complete.")