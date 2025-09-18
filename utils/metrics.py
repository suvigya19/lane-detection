import numpy as np
from shapely.geometry import LineString

def compute_metrics(gt, pred, config, distance_threshold=20):
    """
    Compute precision, recall, and F1 score for lane detection.

    Args:
        gt (list): Ground-truth lanes (list of dictionaries with 'points')
        pred (list): Predicted lanes (list of dictionaries with 'points')
        config (dict): Config dictionary (currently not used, future extensibility)
        distance_threshold (float): Maximum distance in pixels to consider a match.

    Returns:
        dict: Dictionary with 'precision', 'recall', 'f1_score'
    """

    if not pred and not gt:
        return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}
    if not pred and gt:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    if pred and not gt:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    # Convert lanes to LineStrings
    gt_lines = [LineString(lane['points']) for lane in gt if len(lane['points']) >= 2]
    pred_lines = [LineString(lane['points']) for lane in pred if len(lane['points']) >= 2]

    # Initialize counts
    TP = 0
    FP = 0
    FN = 0

    matched_gt = set()

    for pred_idx, pred_line in enumerate(pred_lines):
        matched = False
        for gt_idx, gt_line in enumerate(gt_lines):
            if gt_idx in matched_gt:
                continue  # already matched

            # Compute minimum distance between predicted and GT lane
            distance = pred_line.distance(gt_line)

            if distance < distance_threshold:
                TP += 1
                matched_gt.add(gt_idx)
                matched = True
                break

        if not matched:
            FP += 1

    FN = len(gt_lines) - len(matched_gt)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
