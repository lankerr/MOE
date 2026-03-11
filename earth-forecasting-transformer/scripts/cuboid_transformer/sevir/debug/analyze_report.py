import json, glob, os

files = sorted(
    glob.glob(r'c:\Users\97290\Desktop\MOE\earth-forecasting-transformer\scripts\cuboid_transformer\sevir\debug\nan_report_*.json'),
    key=os.path.getmtime, reverse=True
)
data = json.load(open(files[0]))

# 只看 step 101 的前向事件，找NaN传播链
print("=== Step 101 Forward NaN events ===")
seen = set()
for e in data['all_nan_events']:
    if e['step'] == 101 and 'forward' in e['stage']:
        key = (e['module'], e['stage'])
        if key not in seen:
            seen.add(key)
            mod = e['module']
            stg = e['stage']
            dt = e['dtype']
            nc = e['nan_count']
            tot = e['total_elements']
            stats = e['finite_stats']
            print(f"  {mod} | {stg} | {dt} | nan={nc}/{tot} | stats={stats}")
            if len(seen) >= 25:
                break

print()
print("=== Step 100 events (step before NaN) ===")
events_100 = [e for e in data['all_nan_events'] if e['step'] == 100]
print(f"  Count: {len(events_100)}")
if events_100:
    for e in events_100[:5]:
        print(f"  {e['module']} | {e['stage']} | nan={e['nan_count']}")

print()
print("=== NaN events by step ===")
step_counts = {}
for e in data['all_nan_events']:
    s = e['step']
    step_counts[s] = step_counts.get(s, 0) + 1
for s in sorted(step_counts.keys()):
    print(f"  Step {s}: {step_counts[s]} events")
