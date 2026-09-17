import json
from typing import Any

from backend.extraction.pipeline import KnowledgeGraphExtractionPipeline
from backend.extraction.inspector import SegmentInspection
from backend.extraction.feedback import SegmentFeedback
from backend.extraction.qualifier import (
    ExtractedQualification,
    WorkerQualifications,
)
from backend.extraction.refiner import (
    ActionRefinement,
    SegmentRefinement,
)
from backend.extraction.resolver import ClassifiedObject, SemanticResolution
from backend.schemas.process_knowledge.entities import (
    PPE,
    Procedure,
    ProcessParameter,
    Step,
    Worker,
)
from backend.schemas.process_knowledge.relations import (
    PPERequirement,
    ProcedureExecution,
    ProcedureHasStep,
    QualificationRecord,
    StepOrder,
)
from backend.schemas.process_knowledge.text import (
    Instruction,
    ObservedAction,
    ObservedActor,
    ObservedObject,
    Segment,
)


class PipelineExtractor:
    def extract(self, **kwargs: Any) -> Any:
        response_model = kwargs["response_model"]
        if response_model is SemanticResolution:
            labels = json.loads(kwargs["text"])
            return SemanticResolution.model_validate(
                {
                    "groups": [
                        {"canonical_entity": label, "entities": [label]}
                        for label in labels
                    ]
                }
            )
        if response_model is SegmentInspection:
            return SegmentInspection()
        if response_model is SegmentFeedback:
            return SegmentFeedback()
        if response_model is WorkerQualifications:
            return WorkerQualifications(
                qualifications=[
                    ExtractedQualification(
                        expertise_level="skilled",
                        procedure_domain="inspection",
                    )
                ]
            )
        assert response_model is SegmentRefinement
        return SegmentRefinement(
            actions=[
                ActionRefinement(
                    action_id="action-1",
                    process_parameters=[
                        ProcessParameter(
                            name="Inspection distance",
                            parameter_type="distance",
                            unit="cm",
                            nominal_value=20,
                        )
                    ],
                )
            ]
        )

    def extract_list(self, **kwargs: Any) -> list[Any]:
        item_model = kwargs["item_model"]
        if item_model is Worker:
            return [Worker(name="operator", role="operator")]

        label = json.loads(kwargs["text"])[0]
        return [
            ClassifiedObject(
                input_label=label,
                entity_type="ppe",
                ppe=PPE(name="safety glasses", ppe_type="eye protection"),
            )
        ]


def test_pipeline_extracts_complete_editable_graph() -> None:
    actor = ObservedActor(id="actor-1", name="operator")
    glasses = ObservedObject(
        id="object-1",
        name="safety glasses",
        object_type="protective equipment",
    )
    instruction = Instruction(
        id="instruction-1",
        name="Inspect the assembly",
        segments=[
            Segment(
                id="segment-1",
                scene="workbench",
                actions=[
                    ObservedAction(
                        id="action-1",
                        start_time_ms=0,
                        end_time_ms=1000,
                        actor_id=actor.id,
                        action="Inspect the assembly",
                        instrument_id=glasses.id,
                    ),
                    ObservedAction(
                        id="action-2",
                        start_time_ms=1000,
                        end_time_ms=2000,
                        actor_id=actor.id,
                        action="Document the result",
                        instrument_id=glasses.id,
                    ),
                ],
                actors=[actor],
                objects=[glasses],
                uncertainties=[],
            )
        ],
    )

    graph = KnowledgeGraphExtractionPipeline(
        extractor=PipelineExtractor(),
    ).extract(instruction)

    assert graph.procedure is not None
    assert graph.procedure.procedure_id == "instruction-1"
    assert tuple(graph.steps_by_action_id) == ("action-1", "action-2")
    assert sum(isinstance(entity, Procedure) for entity in graph.entities) == 1
    assert sum(isinstance(entity, Step) for entity in graph.entities) == 2
    assert sum(isinstance(entity, Worker) for entity in graph.entities) == 1
    assert sum(isinstance(entity, PPE) for entity in graph.entities) == 1
    assert sum(
        isinstance(entity, ProcessParameter) for entity in graph.entities
    ) == 1
    assert sum(
        isinstance(relation, PPERequirement) for relation in graph.relations
    ) == 2
    assert sum(
        isinstance(relation, ProcedureHasStep) for relation in graph.relations
    ) == 2
    assert sum(isinstance(relation, StepOrder) for relation in graph.relations) == 1
    assert sum(
        isinstance(relation, ProcedureExecution) for relation in graph.relations
    ) == 1
    assert tuple(graph.process_parameters_by_action_id) == ("action-1",)
    graph.validate()


def test_pipeline_optionally_extracts_worker_qualifications() -> None:
    actor = ObservedActor(id="actor-1", name="operator")
    instruction = Instruction(
        id="instruction-1",
        name="Inspect the assembly",
        segments=[
            Segment(
                id="segment-1",
                scene="workbench",
                actions=[
                    ObservedAction(
                        id="action-1",
                        start_time_ms=0,
                        end_time_ms=1000,
                        actor_id=actor.id,
                        action="Inspect the assembly",
                    )
                ],
                actors=[actor],
                objects=[],
                uncertainties=[],
            )
        ],
    )

    graph = KnowledgeGraphExtractionPipeline(
        extractor=PipelineExtractor(),
        extract_qualifications=True,
        qualification_segments_per_worker=1,
    ).extract(instruction)

    records = [
        relation
        for relation in graph.relations
        if isinstance(relation, QualificationRecord)
    ]
    assert len(records) == 1
    assert records[0].worker.name == "operator"
    assert records[0].procedure_domain == "inspection"
