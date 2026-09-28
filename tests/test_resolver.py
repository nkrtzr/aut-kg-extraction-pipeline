from backend.extraction.resolver import ActorResolver, ObjectResolver
from backend.schemas.process_knowledge.entities import Material, Tool, Worker
from backend.schemas.process_knowledge.text import Instruction, ObservedAction, ObservedActor, ObservedObject, Segment


def test_actor_workers_are_deterministic() -> None:
    actor = ObservedActor(id="actor", name="operator")
    resolver = ActorResolver()
    workers = resolver.create_workers([actor])
    assert workers == [Worker(name="operator", role="operator")]
    assert resolver.worker_bindings == ((actor, workers[0]),)


def test_object_entities_keep_source_name_and_map_assembly_to_material() -> None:
    tool = ObservedObject(id="tool", name="lighter", object_type="tool")
    assembly = ObservedObject(id="assembly", name="wire connection", object_type="assembly")
    resolver = ObjectResolver()
    entities = resolver.create_entities([tool, assembly])
    assert entities == [Tool(name="lighter", tool_type="tool"), Material(name="wire connection", material_type="assembly")]
    assert resolver.entity_bindings[1][0] is assembly


def test_object_resolver_exactly_deduplicates_source_objects() -> None:
    object_ = ObservedObject(id="one", name="wire", object_type="material")
    duplicate = ObservedObject(id="two", name="wire", object_type="material")
    actor = ObservedActor(id="actor", name="operator")
    action = ObservedAction(id="action", start_time_ms=0, end_time_ms=1, actor_id=actor.id, action="work", object_id=object_.id)
    instruction = Instruction(id="i", name="i", segments=[Segment(id="one", scene="", actions=[action], actors=[actor], objects=[object_], uncertainties=[]), Segment(id="two", scene="", actions=[], actors=[], objects=[duplicate], uncertainties=[])])
    assert len(ObjectResolver(instruction).objects) == 1
