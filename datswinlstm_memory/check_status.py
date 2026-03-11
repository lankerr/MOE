import json, os
ckpt = r'c:\Users\97290\Desktop\MOE\datswinlstm_memory\checkpoints'
exps = ['exp12_balanced_moe_rope_flash','exp11_swiglu_moe_rope_flash','exp10_moe_rope_flash','exp9_balanced_moe_flash','exp8_swiglu_moe_flash','exp7_moe_flash']
with open('status_report.txt','w') as out:
    for e in exps:
        lp = os.path.join(ckpt, e, 'training_log.json')
        if os.path.exists(lp):
            with open(lp) as f: data = json.load(f)
            mx = max(d['epoch'] for d in data)
            bv = min(d['val_loss'] for d in data)
            out.write(f"{e}: {len(data)} ep, max={mx}, best_val={bv:.4f}\n")
        else:
            out.write(f"{e}: NO LOG\n")
    out.write(f"384x384: best_model.pth={os.path.exists(os.path.join(ckpt,'384x384','best_model.pth'))}\n")
print(open('status_report.txt').read())
