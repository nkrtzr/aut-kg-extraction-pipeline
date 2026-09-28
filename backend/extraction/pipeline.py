"""Top-level workflow for extracting a procedural knowledge graph."""

from backend.extraction.extractor import Extractor
from backend.extraction.executor import ProcedureExecutionAssembler
from backend.extraction.feedback import FeedbackDiagnostic, FeedbackResolver
from backend.extraction.inspector import Inspector
from backend.extraction.qualifier import QualificationResolver
from backend.extraction.refiner import Refiner
from backend.extraction.resolver import ActorResolver, ObjectResolver
from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.pkg import ProceduralKnowledgeGraph
from backend.schemas.process_knowledge.text import Instruction


class KnowledgeGraphExtractionPipeline:
    """Coordinate all stages required to build a procedure specification KG."""

    def __init__(
        self,
        *,
        extractor: Extractor | None = None,
        extract_qualifications: bool = False,
        qualification_segments_per_worker: int | None = None,
    ) -> None:
        self.extractor = extractor or Extractor()
        self.extract_qualifications = extract_qualifications
        self.qualification_segments_per_worker = qualification_segments_per_worker
        self.feedback_diagnostics: tuple[FeedbackDiagnostic, ...] = ()

    def extract(self, instruction: Instruction) -> ProceduralKnowledgeGraph:
        """Extract and assemble one complete editable specification graph.

        The returned graph contains every entity and relation currently
        produced by the actor resolver, object resolver, sequencer, and
        refiner. Callers do not need to transfer stage outputs manually.
        """

        actor_resolver = ActorResolver(instruction)
        actors = list(actor_resolver.actors)
        workers = actor_resolver.create_workers(actors)

        feedback_resolver = FeedbackResolver(
            instruction,
            extractor=self.extractor,
        )
        self.feedback_diagnostics = feedback_resolver.diagnostics

        qualification_resolver = None
        if self.extract_qualifications:
            qualification_resolver = QualificationResolver(
                instruction,
                actor_resolver.worker_bindings,
                extractor=self.extractor,
                segments_per_worker=self.qualification_segments_per_worker,
            )

        object_resolver = ObjectResolver(instruction)
        objects = list(object_resolver.objects)
        physical_entities = object_resolver.create_entities(
            objects,
            extractor=self.extractor,
        )

        sequencer = Sequencer(
            instruction,
            excluded_action_ids=set(feedback_resolver.feedback_action_ids),
        )
        inspector = Inspector(
            sequencer,
            actor_resolver.worker_bindings,
            extractor=self.extractor,
        )
        procedure_execution = ProcedureExecutionAssembler(
            sequencer.procedure,
            inspector.step_executions,
            quality_checks=inspector.quality_checks,
            feedback=[],
        )
        refiner = Refiner(
            sequencer,
            object_resolver,
            extractor=self.extractor,
        )

        return ProceduralKnowledgeGraph.from_results(
            entities=[
                *workers,
                *physical_entities,
                *inspector.quality_requirements,
            ],
            procedure=sequencer.procedure,
            steps_by_action_id=sequencer.steps_by_action_id,
            process_parameters_by_action_id=(
                refiner.process_parameters_by_action_id
            ),
            relations=[
                *sequencer.procedure_has_steps,
                *refiner.relations,
                *inspector.step_executions,
                *inspector.quality_checks,
                procedure_execution.procedure_execution,
                *(
                    qualification_resolver.qualification_records
                    if qualification_resolver is not None
                    else []
                ),
            ],
        )
