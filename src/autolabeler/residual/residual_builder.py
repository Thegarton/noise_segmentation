def build_masks(num_points: int, actor_indices: list[int], irregular_indices: list[int]) -> dict:
    removed_by_actor = [False] * num_points
    removed_by_irregular = [False] * num_points
    for i in actor_indices:
        removed_by_actor[i] = True
    for i in irregular_indices:
        removed_by_irregular[i] = True
    unexplained_residual = [not (a or b) for a, b in zip(removed_by_actor, removed_by_irregular)]
    return {
        "removed_by_actor": removed_by_actor,
        "removed_by_irregular": removed_by_irregular,
        "unexplained_residual": unexplained_residual,
    }
