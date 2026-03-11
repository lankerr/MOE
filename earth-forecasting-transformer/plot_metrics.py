import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_metrics():
    log_dir = "scripts/cuboid_transformer/sevir/experiments/sevir_rtx5070_run/lightning_logs"
    
    # Get all event files
    event_files = glob.glob(os.path.join(log_dir, "version_*", "events.out.tfevents.*"))
    # Sort them by modification time to get chronological order (handles resume)
    event_files.sort(key=os.path.getmtime)
    
    train_losses = []
    val_losses = []
    train_steps = []
    val_steps = []
    
    for e_file in event_files:
        try:
            ea = EventAccumulator(e_file, size_guidance={'scalars': 0})
            ea.Reload()
            
            # Extract train_loss_epoch if available
            if 'train_loss_epoch' in ea.Tags()['scalars']:
                events = ea.Scalars('train_loss_epoch')
                for e in events:
                    train_steps.append(e.step)
                    train_losses.append(e.value)
                    
            # Extract valid_loss_epoch if available
            if 'valid_loss_epoch' in ea.Tags()['scalars']:
                events = ea.Scalars('valid_loss_epoch')
                for e in events:
                    val_steps.append(e.step)
                    val_losses.append(e.value)
        except Exception as ex:
            print(f"Error reading {e_file}: {ex}")

    if not train_losses and not val_losses:
        print("No metrics found in logs.")
        return

    # Sort in case of overlapping steps (due to resume/crashes)
    if train_losses:
        train_steps, train_losses = zip(*sorted(zip(train_steps, train_losses)))
    if val_losses:
        val_steps, val_losses = zip(*sorted(zip(val_steps, val_losses)))

    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(train_steps, train_losses, label='Train Loss (Epoch)', color='blue', alpha=0.7)
    if val_losses:
        plt.plot(val_steps, val_losses, label='Validation Loss (Epoch)', color='orange', marker='o')

    plt.title('Earthformer SEVIR Training Loss vs Step/Epoch')
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    out_path = "training_loss_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Plot successfully saved to {os.path.abspath(out_path)}")

if __name__ == "__main__":
    plot_metrics()
