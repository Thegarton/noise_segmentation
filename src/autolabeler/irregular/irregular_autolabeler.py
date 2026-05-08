from ..data.schemas import LabelInstance, SequenceSample


class IrregularAutoLabeler:
    def run(self, sample: SequenceSample, removed_by_actor: list[bool]) -> list[LabelInstance]:
        return []
