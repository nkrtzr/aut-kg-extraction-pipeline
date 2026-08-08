"""Top-level workflow for extracting a procedural knowledge graph."""

from backend.extraction.extractor import Extractor
from backend.extraction.refiner import Refiner
from backend.extraction.resolver import ActorResolver, ObjectResolver
from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.pkg import ProceduralKnowledgeGraph
from backend.schemas.process_knowledge.text import Instruction


class KnowledgeGraphExtractionPipeline:
    """Coordinate all stages required to build a procedure specification KG."""

    def __init__(self, *, extractor: Extractor | None = None) -> None:
        self.extractor = extractor or Extractor()

    def extract(self, instruction: Instruction) -> ProceduralKnowledgeGraph:
        """Extract and assemble one complete editable specification graph.

        The returned graph contains every entity and relation currently
        produced by the actor resolver, object resolver, sequencer, and
        refiner. Callers do not need to transfer stage outputs manually.
        """

        actor_resolver = ActorResolver(instruction)
        actors = actor_resolver.semantic_deduplicate(extractor=self.extractor)
        workers = actor_resolver.create_workers(actors, extractor=self.extractor)

        object_resolver = ObjectResolver(instruction)
        objects = object_resolver.semantic_deduplicate(extractor=self.extractor)
        physical_entities = object_resolver.create_entities(
            objects,
            extractor=self.extractor,
        )

        sequencer = Sequencer(instruction)
        refiner = Refiner(
            sequencer,
            object_resolver,
            extractor=self.extractor,
        )

        return ProceduralKnowledgeGraph.from_results(
            entities=[*workers, *physical_entities],
            procedure=sequencer.procedure,
            steps_by_action_id=sequencer.steps_by_action_id,
            process_parameters_by_action_id=(
                refiner.process_parameters_by_action_id
            ),
            relations=refiner.relations,
        )
