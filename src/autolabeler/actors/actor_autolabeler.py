from ..data.schemas import LabelInstance, SequenceSample, Box3D


class ActorAutoLabeler:
    FIXED_CLASSES = ["TRUCK_BUS", "CAR", "CYCLIST", "MOTORCYCLE", "PEDESTRIAN"]

    def run(self, sample: SequenceSample) -> list[LabelInstance]:
        # Placeholder v0 output.
        return [
            LabelInstance(
                frame_id=sample.current.frame_id,
                semantic_class="CAR",
                instance_id=1,
                track_id=1,
                point_indices=[],
                range_image_indices=[],
                box_3d=Box3D(center=[0.0, 0.0, 0.0], size=[4.2, 1.8, 1.6], yaw=0.0, box_type="fixed_actor"),
                mask_confidence=0.6,
                class_confidence=0.7,
                box_confidence=0.6,
                final_confidence=0.63,
                provenance="student_predicted",
                branch_name="actor",
                teacher_sources=[],
                review_status="needs_review",
                pseudo_label_version="v0",
            )
        ]
