import os
import pandas as pd
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
        
        step_to_epoch = {}
        if 'epoch' in tags:
            epoch_events = ea.Scalars('epoch')
            for e in epoch_events:
                step_to_epoch[e.step] = e.value
                
        current_epoch = 0
        epoch_losses = defaultdict(list)
        
        for e in train_events:
            if e.step in step_to_epoch:
                current_epoch = step_to_epoch[e.step]
            ep_int = int(current_epoch)
            epoch_losses[ep_int].append(e.value)
            
        epochs = sorted(epoch_losses.keys())
        avg_losses = [sum(epoch_losses[ep]) / len(epoch_losses[ep]) for ep in epochs]
        return epochs, avg_losses
    except Exception as e:
        pass
    return [], []

def create_table(base_dir, experiments, outfile_csv, outfile_md):
    all_data = {} # {epoch: {exp_name: loss}}
    
    for exp_name, display_name in experiments.items():
        exp_dir = os.path.join(base_dir, exp_name, "lightning_logs")
        if not os.path.exists(exp_dir):
            continue
            
        versions = [d for d in os.listdir(exp_dir) if d.startswith("version_")]
        if not versions:
            continue
            
        versions.sort(key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
        
        all_epoch_losses = defaultdict(list)
        
        for version in versions:
            log_dir = os.path.join(exp_dir, version)
            epochs, avg_losses = get_loss_curve_by_epoch(log_dir)
            if epochs is not None:
                for ep, val in zip(epochs, avg_losses):
                    all_epoch_losses[ep].append(val)
            
        if len(all_epoch_losses) > 0:
            final_epochs = sorted(all_epoch_losses.keys())
            # For each epoch average over versions
            for ep in final_epochs:
                if ep not in all_data:
                    all_data[ep] = {}
                
                # Check if we have data for this epoch to avoid empty division
                if len(all_epoch_losses[ep]) > 0:
                    all_data[ep][display_name] = round(sum(all_epoch_losses[ep])/len(all_epoch_losses[ep]), 6)
                else:
                    all_data[ep][display_name] = pd.NA

    if not all_data:
        print("No valid data found to create table.")
        return

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(all_data, orient='index')
    df.index.name = 'Epoch'
    df.sort_index(inplace=True)
    
    # Save CSV
    df.to_csv(outfile_csv)
    print(f"Saved tabular data to {outfile_csv}")
    
    # Save Markdown
    with open(outfile_md, 'w', encoding='utf-8') as f:
        f.write("# Training Loss by Epoch\n\n")
        f.write(df.to_markdown())
    print(f"Saved markdown data to {outfile_md}")

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')
    
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
    
    create_table(base_dir, exp_20f, "loss_table_20f.csv", "loss_table_20f.md")
