import os
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator

def get_loss_curve_by_epoch(log_dir):
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        
        if 'train_loss' not in tags:
            return None, None
            
        train_events = ea.Scalars('train_loss')
        
        # If 'epoch' is available, map steps to epoch
        step_to_epoch = {}
        if 'epoch' in tags:
            epoch_events = ea.Scalars('epoch')
            for e in epoch_events:
                step_to_epoch[e.step] = e.value
                
        # Group train losses by epoch
        current_epoch = 0
        epoch_losses = defaultdict(list)
        
        for e in train_events:
            if e.step in step_to_epoch:
                current_epoch = step_to_epoch[e.step]
            # Use int(current_epoch) to group nicely (e.g. 0.0, 1.0)
            ep_int = int(current_epoch)
            epoch_losses[ep_int].append(e.value)
            
        # Calculate average per epoch
        epochs = sorted(epoch_losses.keys())
        avg_losses = [sum(epoch_losses[ep]) / len(epoch_losses[ep]) for ep in epochs]
        return epochs, avg_losses
    except Exception as e:
        print(f"Error reading {log_dir}: {e}")
    return [], []

def plot_experiments(base_dir, experiments, title, outfile):
    plt.figure(figsize=(12, 7))
    found_any = False
    
    for exp_name, display_name in experiments.items():
        exp_dir = os.path.join(base_dir, exp_name, "lightning_logs")
        if not os.path.exists(exp_dir):
            print(f"Skipping {exp_name}: log dir not found")
            continue
            
        versions = [d for d in os.listdir(exp_dir) if d.startswith("version_")]
        if not versions:
            print(f"Skipping {exp_name}: no versions found")
            continue
            
        versions.sort(key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
        
        # Dictionary to accumulate all losses across all versions by epoch
        all_epoch_losses = defaultdict(list)
        
        for version in versions:
            log_dir = os.path.join(exp_dir, version)
            epochs, avg_losses = get_loss_curve_by_epoch(log_dir)
            if epochs is not None:
                for ep, val in zip(epochs, avg_losses):
                    all_epoch_losses[ep].append(val)
            
        if len(all_epoch_losses) > 0:
            final_epochs = sorted(all_epoch_losses.keys())
            # For each epoch, if there are multiple versions (e.g., resumed), take the last run's value or average them.
            # Usually we take the average or the last one. We'll average them here:
            final_vals = [sum(all_epoch_losses[ep])/len(all_epoch_losses[ep]) for ep in final_epochs]
            
            plt.plot(final_epochs, final_vals, label=display_name, alpha=0.9, marker='o', linewidth=2.0)
            print(f"Loaded {exp_name} ({display_name}): {len(final_epochs)} epochs from {len(versions)} versions")
            found_any = True
        else:
            print(f"Skipping {exp_name}: train_loss not found in any versions")
            
    if not found_any:
        print(f"No valid logs found for {title}. Skipping plot.")
        plt.close()
        return

    plt.title(title, fontsize=15)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Average Training Loss (train_loss)', fontsize=13)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure x-axis has integer ticks if epochs are small
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved plot to {outfile}")
    plt.close()

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')
    
    # 20-frame experiments
    exp_20f = {
        "exp_earthformer_20frame_mae_mse": "EF-Baseline (20f)",
        "exp_earthformer_exp1_moe_flash": "EF-Exp1 (20f)",
        "exp_earthformer_exp2_swiglu_moe_flash": "EF-Exp2 (20f)",
        "exp_earthformer_exp3_balanced_moe_flash": "EF-Exp3 (20f)",
        "exp_earthformer_exp4_moe_rope_flash": "EF-Exp4 (20f)",
        "exp_earthformer_exp5_swiglu_moe_rope_flash": "EF-Exp5 (20f)",
        "exp_earthformer_exp6_balanced_moe_rope_flash": "EF-Exp6 (20f)",
        "exp_earthformer_exp1_5_moe_balanced_flash": "EF-Exp1.5 (20f)",
    }
    
    # 49-frame experiments
    exp_49f = {
        "exp_earthformer_49f_baseline": "EF-Baseline (49f)",
        "exp_earthformer_49f_exp1_moe_flash": "EF-Exp1 (49f)",
        "exp_earthformer_49f_exp1_5_moe_balanced_flash": "EF-Exp1.5 (49f)",
    }
    
    print("--- Processing 20-frame experiments ---")
    plot_experiments(
        base_dir, 
        exp_20f, 
        "Earthformer 20-frame Training Loss Comparison (By Epoch)", 
        "loss_curve_20f_epoch.png"
    )
    
    print("\n--- Processing 49-frame experiments ---")
    plot_experiments(
        base_dir, 
        exp_49f, 
        "Earthformer 49-frame Training Loss Comparison (By Epoch)", 
        "loss_curve_49f_epoch.png"
    )
