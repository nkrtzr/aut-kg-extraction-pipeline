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
        """Extract and assemble one editable procedural knowledge graph."""

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

        graph = ProceduralKnowledgeGraph()
        graph.add_entities(workers)
        graph.add_entities(physical_entities)
        graph.add_sequence(sequencer.procedure, sequencer.steps_by_action_id)
        for action_id, parameters in refiner.process_parameters_by_action_id.items():
            graph.add_process_parameters(action_id, parameters)
        graph.add_relations(refiner.relations)
        graph.validate()
        return graph
