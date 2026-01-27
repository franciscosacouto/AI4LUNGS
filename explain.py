import torch
from torchsurv.loss import weibull
import numpy as np
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients, NoiseTunnel, visualization as viz

def run_gradient_explanation(lightning_model, dataloader, fold_id):
    # 1. Prepare model
    lightning_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_model.to(device)

    # 2. Get a single batch of test data
    batch = next(iter(dataloader))
    imgs, events, times = batch
    
    # We explain the first image in the batch
    input_tensor = imgs[0:1].to(device)
    input_tensor.requires_grad = True
    
    # 5-year normalized time target
    eval_time = torch.tensor([1825.0 / 2786.0]).to(device)

    
    # 4. Integrated Gradients + Noise Tunnel (Smoothing)
    ig = IntegratedGradients(lightning_model)
    nt = NoiseTunnel(ig)
    
    # Attribute: adds noise 10 times and averages (SmoothGrad)
    attributions = nt.attribute(input_tensor, nt_samples=5, nt_type='smoothgrad_sq', stdevs=0.02, internal_batch_size=1)

    # 5. Visualize and Save
    img_np = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    attr_np = attributions.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    fig, _ = viz.visualize_image_attr_multiple(
        attr_np, img_np,
        ["original_image", "heat_map"],
        ["all", "positive"], # Focus on features increasing risk
        show_plot=False
    )
    
    fig.savefig(f"explanation_fold_{fold_id}.png")
    plt.close(fig)