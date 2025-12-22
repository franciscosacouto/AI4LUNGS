import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os
import json

class VisualizationUploader:
    def __init__(self, client_id: str, album_id: str = None):
        self.client_id = client_id
        self.album_id = album_id  # DeleteHash for the album if needed
        self.headers = {'Authorization': f'Client-ID {self.client_id}'}

    def upload_image(self, image: np.ndarray, file_name: str, dataset_name: str):
        """
        Uploads a single grayscale NumPy array as an image to Imgur.

        Parameters:
            image: NumPy 2D array (single image to upload)
            file_name: Title of the uploaded image
            dataset_name: Description context
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a NumPy ndarray.")

        # Create figure and write image to memory
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image, cmap='gray')
        ax.set_title(file_name)
        ax.axis('off')
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        # Prepare upload data
        data = {
            'album': self.album_id if self.album_id else '',
            'type': 'image/png',
            'title': file_name,
            'description': f"Auto-uploaded from {dataset_name}"
        }

        # Upload to Imgur
        
        try:
            response = requests.post(
                url="https://api.imgur.com/3/upload",
                headers=self.headers,
                files={'image': buffer.getvalue()},
                data=data
            )
            response.raise_for_status()
            print(f"✅ Upload successful: {file_name}")
        except requests.exceptions.RequestException as e:
            print("")