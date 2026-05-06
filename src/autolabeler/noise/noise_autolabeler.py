from ..data.schemas import LabelInstance, SequenceSample


class NoiseAutoLabeler:
    def run(self, sample: SequenceSample, residual_mask: list[bool]) -> list[LabelInstance]:
        return []
