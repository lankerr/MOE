import torch
import argparse
import os
import sys

# Add parent directory to path to handle imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.DATSwinLSTM_D_Memory import Memory
from visualization_utils import set_visualization, print_model_tree

class DummyArgs:
    def __init__(self):
        # Default parameters based on SEVIR/Model defaults
        self.input_img_size = [384, 384] # Reduced for memory safety on local test, usually 384
        self.patch_size = 4
        self.input_channels = 1
        self.embed_dim = 96
        self.depths_down = [2, 2, 2, 2] # Dummy depths
        self.depths_up = [2, 2, 2, 2]
        self.heads_number = [3, 6, 12, 24]
        self.window_size = 7
        self.out_len = 12 # Predict 12 frames
        
def main():
    print("Initializing Model...")
    args = DummyArgs()
    
    # Initialize model
    # Note: Using smaller memory sizes to ensure it runs on variety of hardware for testing
    model = Memory(args, memory_channel_size=96, memory_slot_size=10, short_len=5, long_len=10)
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Generate Model Tree
    print("Generating Model Tree...")
    with open("results/visualization/model_tree.txt", "w") as f:
        f.write("DATSwinLSTM Model Structure:\n")
        print_model_tree(model, file=f)
    print("Model Tree saved to results/visualization/model_tree.txt")

    # Prepare dummy input
    # standard input shape: (B, T, C, H, W)
    # Memory.forward takes: inputs (B, T, C, H, W), memory_x (B, T_mem, C, H, W), phase
    B = 1
    T_input = 5
    T_memory = 5 
    C = 1
    H, W = 384, 384
    
    inputs = torch.randn(B, T_input, C, H, W).to(device)
    memory_x = torch.randn(B, T_memory, C, H, W).to(device)
    phase = 0 # 0 or 1
    
    print("Running Forward Pass with Visualization...")
    set_visualization(True, save_dir="results/visualization")
    
    with torch.no_grad():
        try:
            outputs = model(inputs, memory_x, phase)
            print("Forward pass successful!")
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()

    print(f"Visualizations saved to results/visualization")

    # Memory Analysis (Approximate)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model Parameters Size: {param_size / 1024**2:.2f} MB")
    
    # Estimated Memory for Batch Size 4, 384x384 (very rough)
    # 4 * (Input + Output + Activations)
    # This is complex to calculate exactly without running on GPU and measuring peak.
    print("\n--- Compute Resource Analysis ---")
    print(f"Input Tensor Shape: {inputs.shape}")
    print("Single GTX 3090 (24GB) Status: Likely Sufficient for Inference/Training with small batch size (1-4).")
    print("For full 100GB dataset training, use SCOW AI Platform with distributed training.")

if __name__ == "__main__":
    if not os.path.exists("results/visualization"):
        os.makedirs("results/visualization")
    main()
