def compute_final_confidence(mask_conf: float, class_conf: float, box_conf: float, temporal_consistency: float = 0.5) -> float:
    weights = (0.35, 0.35, 0.2, 0.1)
    score = mask_conf * weights[0] + class_conf * weights[1] + box_conf * weights[2] + temporal_consistency * weights[3]
    return max(0.0, min(1.0, score))
