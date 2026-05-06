from dataclasses import dataclass


@dataclass
class ReviewItem:
    frame_id: str
    instance_id: int
    reason: str


def build_review_queue(labels: list, confidence_threshold: float = 0.6) -> list[ReviewItem]:
    queue = []
    for label in labels:
        if label.final_confidence < confidence_threshold or label.review_status == "needs_review":
            queue.append(ReviewItem(frame_id=label.frame_id, instance_id=label.instance_id, reason="low_conf_or_flagged"))
    return queue
