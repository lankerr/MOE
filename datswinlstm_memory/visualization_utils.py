import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime

# Global flag to control visualization
VISUALIZE = False
SAVE_DIR = "results/visualization"

def set_visualization(enabled, save_dir="results/visualization"):
    global VISUALIZE, SAVE_DIR
    VISUALIZE = enabled
    SAVE_DIR = save_dir
    if enabled and not os.path.exists(save_dir):
        os.makedirs(save_dir)

def flow2rgb(flow_map, max_value=None):
    """
    Visualizes optical flow map.
    :param flow_map: (H, W, 2) numpy array
    :return: (H, W, 3) RGB image
    """
    flow_map = np.array(flow_map)
    h, w, _ = flow_map.shape
    flow_map[:,:,0] += np.arange(w)
    flow_map[:,:,1] += np.arange(h)[:,np.newaxis]
    
    # Use Hue for direction, Saturation for magnitude
    rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1])
    
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    if max_value is None:
        max_value = np.max(mag)
    if max_value == 0:
        max_value = 1
        
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def plot_optical_flow(flow, name="flow"):
    """
    Plot and save optical flow.
    flow: tensor describing flow (B, 2, H, W) or (2, H, W)
    """
    if not VISUALIZE:
        return

    if flow.dim() == 4:
        flow = flow[0] # Take first item in batch

    # Permute to (H, W, 2) for visualization
    flow_np = flow.permute(1, 2, 0).detach().cpu().numpy()
    
    # Magnitude and angle
    magnitude = np.sqrt(flow_np[..., 0]**2 + flow_np[..., 1]**2)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Quiver (Arrows) - subsample for clarity
    step = 16
    y, x = np.mgrid[0:flow_np.shape[0]:step, 0:flow_np.shape[1]:step]
    fx = flow_np[::step, ::step, 0]
    fy = flow_np[::step, ::step, 1]
    
    ax[0].quiver(x, y, fx, fy, color='r')
    ax[0].invert_yaxis()
    ax[0].set_title(f"Optical Flow Vectors ({name})")
    
    # Plot 2: Magnitude Heatmap
    im = ax[1].imshow(magnitude, cmap='hot')
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title(f"Flow Magnitude ({name})")
    
    timestamp = datetime.now().strftime("%H%M%S_%f")
    plt.savefig(os.path.join(SAVE_DIR, f"{name}_{timestamp}.png"))
    plt.close()

def plot_deformable_offsets(pos, ref, name="deform_attn"):
    """
    Visualize deformable attention offsets.
    pos: (B, H, W, 2) - Deformed points
    ref: (B, H, W, 2) - Reference points
    """
    if not VISUALIZE:
        return
        
    # Take first sample in batch
    if pos.dim() > 3:
        pos = pos[0]
        ref = ref[0]
        
    pos = pos.detach().cpu().numpy()
    ref = ref.detach().cpu().numpy()
    
    # Flatten for scatter plot
    h, w, _ = pos.shape
    ref_flat = ref.reshape(-1, 2)
    pos_flat = pos.reshape(-1, 2)
    
    # Subsample for clarity
    total_points = ref_flat.shape[0]
    indices = np.random.choice(total_points, min(500, total_points), replace=False)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(ref_flat[indices, 0], ref_flat[indices, 1], c='b', alpha=0.5, label='Reference')
    plt.scatter(pos_flat[indices, 0], pos_flat[indices, 1], c='r', alpha=0.5, label='Deformed')
    
    # Draw lines connecting ref to pos
    for i in indices:
        plt.plot([ref_flat[i, 0], pos_flat[i, 0]], [ref_flat[i, 1], pos_flat[i, 1]], 'k-', alpha=0.1)
        
    plt.title(f"Deformable Attention Offsets ({name})")
    plt.legend()
    plt.gca().invert_yaxis()
    
    timestamp = datetime.now().strftime("%H%M%S_%f")
    plt.savefig(os.path.join(SAVE_DIR, f"{name}_{timestamp}.png"))
    plt.close()

def plot_memory_attention(attn_map, name="memory_attn"):
    """
    Plot attention map heatmaps.
    attn_map: (B, H, W) or (H, W)
    """
    if not VISUALIZE:
        return
        
    if attn_map.dim() == 3:
        attn_map = attn_map[0]
        
    attn_map = attn_map.detach().cpu().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(attn_map, cmap='viridis')
    plt.colorbar()
    plt.title(f"Memory Attention ({name})")
    
    timestamp = datetime.now().strftime("%H%M%S_%f")
    plt.savefig(os.path.join(SAVE_DIR, f"{name}_{timestamp}.png"))
    plt.close()

def print_model_tree(model, indent=0, file=None):
    """
    Recursively print the model structure as a tree.
    """
    if file is None:
        return

    for name, child in model.named_children():
        file.write("  " * indent + f"|-- {name}: {child._get_name()}\n")
        print_model_tree(child, indent + 1, file)
