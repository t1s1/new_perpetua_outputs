import csv
from collections import defaultdict
from datetime import datetime, timezone

path = "/sessions/keen-clever-franklin/mnt/uploads/checkpoint_scan_last.csv"

rows = []
with open(path, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            'step': int(r['step']),
            'checkpoint_path': r['checkpoint_path'],
            'reward': float(r['det_reward_1ep']),
            'angle': float(r['det_peak_angle_deg_1ep']),
            'mtime': float(r['mtime']),
        })

print(f"Total rows loaded: {len(rows)}")

# Dedupe by step, keep latest mtime
by_step = {}
dupe_count = 0
for r in rows:
    s = r['step']
    if s in by_step:
        dupe_count += 1
        if r['mtime'] > by_step[s]['mtime']:
            by_step[s] = r
    else:
        by_step[s] = r

deduped = list(by_step.values())
print(f"Deduped rows: {len(deduped)}  (duplicate-step rows encountered: {dupe_count})")
print(f"Unique steps: {len(by_step)}")

# Show which steps had duplicates
step_counts = defaultdict(int)
for r in rows:
    step_counts[r['step']] += 1
dup_steps = {s:c for s,c in step_counts.items() if c > 1}
print(f"Number of steps with >1 scan entries: {len(dup_steps)}")
if dup_steps:
    print("Example dup steps (step: count):", dict(list(sorted(dup_steps.items()))[:20]))


print("\n" + "="*80)
print("TASK 2: Most recent 60 rows by mtime ascending (chronological order)")
print("="*80)

sorted_by_mtime = sorted(deduped, key=lambda r: r['mtime'])
last60 = sorted_by_mtime[-60:]

def fmt_time(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

print(f"{'idx':>4} {'step':>8} {'reward':>10} {'angle':>8} {'mtime':>18} {'mtime_readable':>22} {'good?':>6}")
for i, r in enumerate(last60):
    good = "GOOD" if 55 <= r['angle'] <= 95 else ""
    print(f"{i:>4} {r['step']:>8} {r['reward']:>10.4f} {r['angle']:>8.4f} {r['mtime']:>18.6f} {fmt_time(r['mtime']):>22} {good:>6}")

print("\n" + "="*80)
print("TASK 3: Runs of 3+ consecutive (mtime-order) checkpoints in [55,95] within last-60 slice")
print("="*80)

runs = []
cur = []
for r in last60:
    if 55 <= r['angle'] <= 95:
        cur.append(r)
    else:
        if len(cur) >= 3:
            runs.append(list(cur))
        cur = []
if len(cur) >= 3:
    runs.append(list(cur))

if not runs:
    print("No run of 3+ consecutive good checkpoints found in the most-recent-60 mtime-ordered slice.")
    # also show all runs of length >=2 for context
    cur = []
    runs2 = []
    for r in last60:
        if 55 <= r['angle'] <= 95:
            cur.append(r)
        else:
            if len(cur) >= 2:
                runs2.append(list(cur))
            cur = []
    if len(cur) >= 2:
        runs2.append(list(cur))
    print(f"\nFor context, runs of length >=2 found in last-60 slice: {len(runs2)}")
    for run in runs2:
        steps = [x['step'] for x in run]
        print(f"  length={len(run)} steps={steps}")
        for x in run:
            print(f"    step={x['step']} reward={x['reward']:.4f} angle={x['angle']:.4f} mtime={x['mtime']:.6f}")
else:
    for run in runs:
        steps = [x['step'] for x in run]
        print(f"Run length={len(run)} step_range=({steps[0]},{steps[-1]})")
        for x in run:
            print(f"  step={x['step']} reward={x['reward']:.4f} angle={x['angle']:.4f} mtime={x['mtime']:.6f}")


print("\n" + "="*80)
print("TASK 4: Lineage-aware streak algorithm (group by step % 50000)")
print("="*80)

groups = defaultdict(list)
for r in deduped:
    offset = r['step'] % 50000
    groups[offset].append(r)

all_runs = []  # (length, offset, run_list)
for offset, items in groups.items():
    items_sorted = sorted(items, key=lambda r: r['step'])
    cur = []
    for r in items_sorted:
        is_good = 55 <= r['angle'] <= 95
        if is_good:
            if cur and r['step'] - cur[-1]['step'] == 50000:
                cur.append(r)
            else:
                # start new run (either first good, or gap != 50000 means not "consecutive")
                cur = [r]
            # check contiguous run so far - but need to record run whenever it might end
        else:
            cur = []
            continue
        # record snapshot of current run state
        all_runs.append((len(cur), offset, list(cur)))

# For each offset+run, we recorded many overlapping snapshots; we just want maximal runs.
# Better: redo cleanly, only append run to results list when it breaks (or at end), keeping only maximal runs.
all_runs = []
for offset, items in groups.items():
    items_sorted = sorted(items, key=lambda r: r['step'])
    cur = []
    for r in items_sorted:
        is_good = 55 <= r['angle'] <= 95
        if is_good and cur and r['step'] - cur[-1]['step'] == 50000:
            cur.append(r)
        elif is_good:
            cur = [r]
        else:
            if len(cur) >= 1:
                all_runs.append((len(cur), offset, list(cur)))
            cur = []
    if cur:
        all_runs.append((len(cur), offset, list(cur)))

# sort all_runs by length descending
all_runs.sort(key=lambda x: -x[0])

print(f"Number of offset-groups (distinct step % 50000 values): {len(groups)}")
print(f"Total maximal good-runs found across all offset-groups: {len(all_runs)}")

print("\n--- LONGEST RUN OVERALL ---")
if all_runs:
    length, offset, run = all_runs[0]
    steps = [x['step'] for x in run]
    print(f"Length: {length}")
    print(f"Offset (step % 50000): {offset}")
    print(f"Step range: {steps[0]} .. {steps[-1]}")
    print("Detail:")
    for x in run:
        print(f"  step={x['step']:>10} reward={x['reward']:>10.4f} angle={x['angle']:>8.4f} mtime={x['mtime']:.6f} ({fmt_time(x['mtime'])})")
else:
    print("No runs found at all.")

print("\n--- TOP 5 LONGEST RUNS ---")
for i, (length, offset, run) in enumerate(all_runs[:5]):
    steps = [x['step'] for x in run]
    angles = [x['angle'] for x in run]
    print(f"#{i+1}: length={length} offset={offset} step_range=({steps[0]},{steps[-1]}) angle_min={min(angles):.4f} angle_max={max(angles):.4f}")


print("\n--- DETAIL FOR ALL RUNS TIED AT MAX LENGTH ---")
max_len = all_runs[0][0]
tied = [x for x in all_runs if x[0] == max_len]
print(f"Number of runs tied at max length {max_len}: {len(tied)}")
for length, offset, run in tied:
    steps = [x['step'] for x in run]
    print(f"\nOffset={offset} step_range=({steps[0]},{steps[-1]})")
    for x in run:
        print(f"  step={x['step']:>10} reward={x['reward']:>10.4f} angle={x['angle']:>8.4f} mtime={x['mtime']:.6f} ({fmt_time(x['mtime'])})")

print("\n" + "="*80)
print("TASK 5: Max step number and its mtime")
print("="*80)
max_row = max(deduped, key=lambda r: r['step'])
print(f"Max step: {max_row['step']}")
print(f"Its reward: {max_row['reward']:.4f}, angle: {max_row['angle']:.4f}")
print(f"Its mtime: {max_row['mtime']:.6f}  ({fmt_time(max_row['mtime'])})")

# also latest mtime row (should generally correspond but confirm)
latest_mtime_row = max(deduped, key=lambda r: r['mtime'])
print(f"\nLatest mtime row: step={latest_mtime_row['step']} mtime={latest_mtime_row['mtime']:.6f} ({fmt_time(latest_mtime_row['mtime'])})")

