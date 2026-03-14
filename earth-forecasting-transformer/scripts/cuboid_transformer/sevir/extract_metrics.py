import os
from tensorboard.backend.event_processing import event_accumulator

def extract_tags(log_dir):
    print(f"Checking {log_dir}...")
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if 'tfevents' in file:
                path = os.path.join(root, file)
                ea = event_accumulator.EventAccumulator(path)
                ea.Reload()
                tags = ea.Tags()
                print(f"File: {path}")
                if 'scalars' in tags:
                    print("  Scalars:", tags['scalars'])

if __name__ == '__main__':
    log_dir = "experiments/exp_earthformer_exp1_moe_flash/lightning_logs/version_0"
    extract_tags(log_dir)
