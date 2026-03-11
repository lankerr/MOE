import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def print_metrics():
    log_dir = "scripts/cuboid_transformer/sevir/experiments/sevir_rtx5070_run/lightning_logs"
    event_files = glob.glob(os.path.join(log_dir, "version_*", "events.out.tfevents.*"))
    event_files.sort(key=os.path.getmtime)
    
    train_metrics = {}
    val_metrics = {}
    
    for f in event_files:
        ea = EventAccumulator(f, size_guidance={'scalars': 0})
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        
        if 'train_loss_epoch' in tags:
            for e in ea.Scalars('train_loss_epoch'):
                train_metrics[e.step] = e.value
                
        if 'valid_loss_epoch' in tags:
            for e in ea.Scalars('valid_loss_epoch'):
                val_metrics[e.step] = e.value
                
    steps = sorted(list(set(train_metrics.keys()) | set(val_metrics.keys())))
    
    with open("losses_dump.txt", "w") as out:
        out.write("Step\tTrain_Loss\tValid_Loss(-CSI)\n")
        for s in steps:
            # only print every 2000 steps or near 40000
            if s % 2000 == 0 or 38000 < s < 45000:
                tr = f"{train_metrics[s]:.5f}" if s in train_metrics else "N/A"
                vl = f"{val_metrics[s]:.5f}" if s in val_metrics else "N/A"
                out.write(f"{s}\t{tr}\t{vl}\n")

if __name__ == "__main__":
    print_metrics()
