from ..data.schemas import LabelInstance


def arbitrate(labels: list[LabelInstance]) -> list[LabelInstance]:
    # v0: keep all, later prioritize by branch conflict logic.
    return labels
