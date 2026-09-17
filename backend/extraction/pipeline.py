"""Top-level workflow for extracting a procedural knowledge graph."""

from backend.extraction.extractor import Extractor
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

    def extract(self, instruction: Instruction) -> ProceduralKnowledgeGraph:
        """Extract and assemble one complete editable specification graph.

        The returned graph contains every entity and relation currently
        produced by the actor resolver, object resolver, sequencer, and
        refiner. Callers do not need to transfer stage outputs manually.
        """

        actor_resolver = ActorResolver(instruction)
        actors = actor_resolver.semantic_deduplicate(extractor=self.extractor)
        workers = actor_resolver.create_workers(actors, extractor=self.extractor)

        qualification_resolver = None
        if self.extract_qualifications:
            qualification_resolver = QualificationResolver(
                instruction,
                actor_resolver.worker_bindings,
                extractor=self.extractor,
                segments_per_worker=self.qualification_segments_per_worker,
            )

        object_resolver = ObjectResolver(instruction)
        objects = object_resolver.semantic_deduplicate(extractor=self.extractor)
        physical_entities = object_resolver.create_entities(
            objects,
            extractor=self.extractor,
        )

        sequencer = Sequencer(instruction)
        inspector = Inspector(
            sequencer,
            actor_resolver.worker_bindings,
            extractor=self.extractor,
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
                *(
                    qualification_resolver.qualification_records
                    if qualification_resolver is not None
                    else []
                ),
            ],
        )
