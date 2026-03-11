import json, sys

path = sys.argv[1]
data = json.load(open(path))

fn = data['first_nan']
print(f"Total steps: {data['total_steps']}, Total NaN events: {data['total_nan_events']}")
if fn:
    print(f"\n=== FIRST NaN ===")
    print(f"  Step: {fn['step']}")
    print(f"  Module: {fn['module']}")
    print(f"  Stage: {fn['stage']}")
    print(f"  dtype: {fn['dtype']}")
    print(f"  shape: {fn['shape']}")
    print(f"  nan: {fn['nan_count']}/{fn['total_elements']}")
    print(f"  inf: {fn['inf_count']}")
    print(f"  stats: {fn['finite_stats']}")

# 找到首次 NaN 的 step
first_step = fn['step'] if fn else None
if first_step is not None:
    print(f"\n=== Step {first_step} Forward NaN (unique modules) ===")
    seen = set()
    for e in data['all_nan_events']:
        if e['step'] == first_step and 'forward' in e['stage']:
            key = e['module']
            if key not in seen:
                seen.add(key)
                print(f"  {e['module']} | {e['stage']} | {e['dtype']} | nan={e['nan_count']}/{e['total_elements']}")
                if 'finite_stats' in e:
                    s = e['finite_stats']
                    if isinstance(s.get('abs_max'), (int, float)):
                        print(f"    -> abs_max={s['abs_max']:.4e}")
                if len(seen) >= 20:
                    break

    # check step before
    prev = first_step - 1
    prev_events = [e for e in data['all_nan_events'] if e['step'] == prev]
    print(f"\n=== Step {prev} events: {len(prev_events)} ===")

# NaN by step
print(f"\n=== NaN events by step ===")
step_counts = {}
for e in data['all_nan_events']:
    s = e['step']
    step_counts[s] = step_counts.get(s, 0) + 1
for s in sorted(step_counts.keys()):
    print(f"  Step {s}: {step_counts[s]} events")
