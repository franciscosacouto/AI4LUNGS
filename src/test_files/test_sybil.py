from huggingface_hub import snapshot_download
import sys
import os
import torch
import numpy as np

# Download and setup model
model_path = snapshot_download(repo_id="Lab-Rasool/sybil")
sys.path.append(model_path)

from modeling_sybil_hf import SybilHFWrapper
from configuration_sybil import SybilConfig

def extract_embeddings(dicom_paths):
    """
    Extract embeddings from the layer after ReLU, before Dropout.

    Args:
        dicom_paths: List of DICOM file paths

    Returns:
        numpy array of shape (512,) - averaged embeddings across ensemble
    """
    # Initialize model
    config = SybilConfig()
    model = SybilHFWrapper(config)

    # Set each model in ensemble to eval mode
    for m in model.models:
        m.eval()

    # Storage for embeddings from each model in ensemble
    all_embeddings = []

    # Register hooks on each model in the ensemble
    for model_idx, ensemble_model in enumerate(model.models):
        embeddings_buffer = []

        def create_hook(buffer):
            def hook(module, input, output):
                # Capture the output of ReLU layer (before dropout)
                buffer.append(output.detach().cpu())
            return hook

        # Register hook on the ReLU layer
        hook_handle = ensemble_model.relu.register_forward_hook(create_hook(embeddings_buffer))

        # Run forward pass
        with torch.no_grad():
            _ = model(dicom_paths=dicom_paths)

        # Remove hook
        hook_handle.remove()

        # Get the embeddings (should be shape [1, 512])
        if embeddings_buffer:
            embedding = embeddings_buffer[0].numpy().squeeze()
            all_embeddings.append(embedding)
            print(f"Model {model_idx + 1}: Embedding shape = {embedding.shape}")

    # Average embeddings across ensemble
    averaged_embedding = np.mean(all_embeddings, axis=0)
    return averaged_embedding

# Usage
dicom_dir = "path/to/volume"
dicom_paths = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]

embeddings = extract_embeddings(dicom_paths)
print(f"\nEmbedding vector shape: {embeddings.shape}")
print(f"Embedding statistics:")
print(f"  Mean: {np.mean(embeddings):.6f}")
print(f"  Std: {np.std(embeddings):.6f}")
print(f"  Min: {np.min(embeddings):.6f}")
print(f"  Max: {np.max(embeddings):.6f}")