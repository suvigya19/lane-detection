import os
import json
import numpy as np
from utils.metrics import compute_metrics
from utils.tokenizer import LaneTokenizer
import yaml
from tqdm import tqdm

# === Load config ===
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# === Paths ===
gt_file = config['evaluation']['ground_truth_dir']  # Single test_label.json file
pred_dir = config['evaluation']['prediction_dir']

# === Initialize tokenizer (future extensibility) ===
tokenizer = LaneTokenizer(nbins=config['vocab_size'] - 7)

# === Load GT annotations ===
print("[Info] Loading ground truth annotations...")
gt_annotations = {}
with open(gt_file, 'r') as f:
    for line in f:
        ann = json.loads(line)
        base_name = os.path.splitext(ann['raw_file'])[0].replace('/', '_')
        gt_annotations[base_name] = {
            'lanes': ann['lanes'],
            'h_samples': ann['h_samples']
        }

print(f"[Info] Total ground truth annotations loaded: {len(gt_annotations)}")

# === Prepare prediction files ===
pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
print(f"[Info] Total prediction files found: {len(pred_files)}")

all_metrics = []

for pred_filename in tqdm(pred_files, desc="Evaluating"):
    pred_path = os.path.join(pred_dir, pred_filename)
    base_name = os.path.splitext(pred_filename)[0]

    # Retrieve GT annotation
    if base_name not in gt_annotations:
        print(f"[Warning] No ground truth found for {base_name}. Skipping.")
        continue

    gt_data = gt_annotations[base_name]
    gt_lanes = gt_data['lanes']
    h_samples = gt_data['h_samples']

    # Convert GT lanes to list of points
    gt = []
    for lane in gt_lanes:
        points = []
        for x, y in zip(lane, h_samples):
            if x >= 0:
                points.append([x, y])
        if len(points) >= 2:
            gt.append({'points': points})

    # Load predictions
    with open(pred_path, 'r') as f:
        pred = json.load(f)

    # Compute metrics for the sample
    metrics = compute_metrics(gt, pred, config)
    all_metrics.append(metrics)

# === Aggregate results ===
if all_metrics:
    precision = np.mean([m['precision'] for m in all_metrics])
    recall = np.mean([m['recall'] for m in all_metrics])
    f1_score = np.mean([m['f1_score'] for m in all_metrics])

    print("\n=== Evaluation Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")

    # Optionally, save results to JSON
    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'samples_evaluated': len(all_metrics)
    }

    output_path = os.path.join(pred_dir, "evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n[Info] Results saved to {output_path}")

else:
    print("[Warning] No valid predictions were evaluated!")
