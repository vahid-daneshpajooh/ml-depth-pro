import torch
import gradio as gr
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import os

from depth_pro import create_model_and_transforms, load_rgb

def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device
    
def predict_depth(image_path, auto_rotate: bool, remove_alpha: bool, model, transform):
    
    # Load and preprocess the image from the given path
    image, _, f_px = load_rgb(image_path, auto_rotate=auto_rotate, remove_alpha=remove_alpha)

    # Run inference
    prediction = model.infer(transform(image), f_px=f_px)
    
    # Extract the depth and focal length.
    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    if f_px is not None:
        print(f"Focal length (from exif): {f_px:0.2f}")
        focallength_px = f_px
    elif prediction["focallength_px"] is not None:
        focallength_px = prediction["focallength_px"].detach().cpu().item()
        print(f"Estimated focal length: {focallength_px: 0.0f}")

    # Visualize inverse depth instead of depth
    inverse_depth = 1 / depth

    # clipped to [0.1m;250m] range for better visualization.
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
    )

    # Normalize and save as color-mapped "turbo" jpg
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)

    # output image
    outImage_path = os.path.split(image_path)[0] + ".png"
    Image.fromarray(color_depth).save(outImage_path, format = "PNG")
    
    # Return depth map and f_px
    return outImage_path, focallength_px

def main():
    
    # Load model and preprocessing transform
    model, transform = create_model_and_transforms() 
    # for gpu usage add args:
    # (Note: currently, does't work)
    # device = get_torch_device(), precision = torch.half)
    model.eval()

    # Set up Gradio interface
    iface = gr.Interface(
        fn=lambda image, auto_rotate, remove_alpha: predict_depth(image, auto_rotate, remove_alpha, model, transform),
        inputs=[
            gr.Image(type="filepath", label="Upload Image"),
            gr.Checkbox(label="Auto Rotate", value=True),  # Checkbox for auto_rotate
            gr.Checkbox(label="Remove Alpha", value=True)   # Checkbox for remove_alpha
        ],
        outputs=[
            gr.Image(label="Depth Map", type = "filepath"),
            gr.Textbox(label="Focal Length in Pixels", placeholder="Focal length")  # Output for f_px
        ],
        title="Depth Pro: Sharp Monocular Metric Depth Estimation",  # Set the title to "Depth Pro"
        description="Upload an image and adjust options to estimate its depth map using Apple Depth-Pro model.",
        allow_flagging=False  # Disable the flag button
    )

    # Launch the interface
    iface.launch(server_name = "0.0.0.0", server_port = 7860)

if __name__ == "__main__":
    main()