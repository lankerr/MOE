import os
import re
import sys
import time
import json
import signal
import subprocess

from config import cfg
from experiments.experiment_factory import EXPERIMENTS


# 按当前实验推进策略: 先保证前 10 个实验可启动
TARGET_EXPS = [
    "exp1_moe",
    "exp2_swiglu_moe",
    "exp3_balanced_moe",
    "exp4_moe_rope",
    "exp5_swiglu_moe_rope",
    "exp6_balanced_moe_rope",
    "exp7_moe_flash",
    "exp8_swiglu_moe_flash",
    "exp9_balanced_moe_flash",
    "exp10_moe_rope_flash",
]


def check_dataset_paths():
    paths = cfg.get_sevir_paths()
    report = {
        "root_dir": paths["root_dir"],
        "catalog_path": paths["catalog_path"],
        "data_dir": paths["data_dir"],
        "missing": [],
    }
    if not os.path.exists(paths["root_dir"]):
        report["missing"].append(paths["root_dir"])
    if not os.path.exists(paths["catalog_path"]):
        report["missing"].append(paths["catalog_path"])
    if not os.path.exists(paths["data_dir"]):
        report["missing"].append(paths["data_dir"])
    return report


def run_until_epoch_start(exp_name, timeout_sec=180):
    cmd = [
        sys.executable,
        "-u",
        "train_experiment.py",
        "--exp", exp_name,
        "--epochs", "1",
        "--num_workers", "0",
        "--batch_size", "1",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    t0 = time.time()
    seen_epoch = False
    last_lines = []
    epoch_pattern = re.compile(r"Epoch\s+1/|Epoch\s+1\b")

    try:
        while True:
            if proc.poll() is not None:
                break

            line = proc.stdout.readline()
            if line:
                line_strip = line.rstrip("\n")
                last_lines.append(line_strip)
                if len(last_lines) > 50:
                    last_lines.pop(0)

                if epoch_pattern.search(line_strip):
                    seen_epoch = True
                    # 一旦进入 epoch 就停止当前实验
                    try:
                        proc.terminate()
                        proc.wait(timeout=8)
                    except Exception:
                        proc.kill()
                    break

            if time.time() - t0 > timeout_sec:
                try:
                    proc.terminate()
                    proc.wait(timeout=8)
                except Exception:
                    proc.kill()
                break

        rc = proc.poll()
        if rc is None:
            rc = -999

        return {
            "exp": exp_name,
            "seen_epoch": seen_epoch,
            "return_code": rc,
            "timeout_sec": timeout_sec,
            "tail": last_lines[-20:],
        }
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass


def main():
    dataset_report = check_dataset_paths()

    results = []
    print(f"[INFO] Target experiments ({len(TARGET_EXPS)}): {', '.join(TARGET_EXPS)}")
    for exp_name in TARGET_EXPS:
        if exp_name not in EXPERIMENTS:
            print(f"\n[CHECK] {exp_name}")
            print("  -> FAIL (experiment not defined)")
            results.append({
                "exp": exp_name,
                "seen_epoch": False,
                "return_code": -2,
                "timeout_sec": 0,
                "tail": ["experiment not defined in EXPERIMENTS"],
            })
            continue

        print(f"\n[CHECK] {exp_name}")
        r = run_until_epoch_start(exp_name, timeout_sec=240)
        status = "PASS" if r["seen_epoch"] else "FAIL"
        print(f"  -> {status} (seen_epoch={r['seen_epoch']}, rc={r['return_code']})")
        results.append(r)

    summary = {
        "dataset": dataset_report,
        "results": results,
        "pass_count": sum(1 for r in results if r["seen_epoch"]),
        "fail_count": sum(1 for r in results if not r["seen_epoch"]),
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_all_train_start_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"PASS: {summary['pass_count']} | FAIL: {summary['fail_count']}")
    print(f"Report: {out_path}")

    if dataset_report["missing"]:
        print("[MISSING]")
        for p in dataset_report["missing"]:
            print(" -", p)


if __name__ == "__main__":
    main()
