"""Schemas for observations extracted from text and video segments."""

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, NewType

from pydantic import BaseModel, Field, model_validator

ActionId = NewType("ActionId", str)
ActorId = NewType("ActorId", str)
ObjectId = NewType("ObjectId", str)
SegmentId = NewType("SegmentId", str)


class ObservedAction(BaseModel):
    """An action extracted from a text or video source.

    Actor and entity references use typed observation IDs. Display names are
    never used as foreign keys. Timestamps are optional for text sources.
    """

    id: ActionId
    start_time_ms: int | None = None
    end_time_ms: int | None = None

    actor_id: ActorId
    action: str
    object_id: ObjectId | None = None
    instrument_id: ObjectId | None = None
    target_id: ObjectId | None = None


class ObservedObject(BaseModel):
    """An object observed in a segment."""

    id: ObjectId
    name: str
    object_type: str
    action_ids: list[ActionId] = Field(default_factory=list)

    def matches(self, other: object) -> bool:
        """Match objects by their observed name and type."""

        return (
            isinstance(other, ObservedObject)
            and self.name == other.name
            and self.object_type == other.object_type
        )

    def merge(self, other: "ObservedObject") -> None:
        """Merge action references while preserving observation order."""

        self.action_ids = list(dict.fromkeys([*self.action_ids, *other.action_ids]))


class ObservedActor(BaseModel):
    """An actor observed in a video segment."""

    id: ActorId
    name: str
    action_ids: list[ActionId] = Field(default_factory=list)

    def matches(self, other: object) -> bool:
        """Match actors by their observed name."""

        return isinstance(other, ObservedActor) and self.name == other.name

    def merge(self, other: "ObservedActor") -> None:
        """Merge action references while preserving observation order."""

        self.action_ids = list(dict.fromkeys([*self.action_ids, *other.action_ids]))


class Segment(BaseModel):
    """A text-derived segment containing observations to be processed."""

    id: SegmentId
    scene: str
    actions: list[ObservedAction]
    actors: list[ObservedActor]
    objects: list[ObservedObject]
    uncertainties: list[str]


class VideoSegment(Segment):
    """A time-bounded segment produced by the video-processing pipeline.

    Repeated object and actor names are numbered after validation so that every
    observed entity has a unique name within the segment.
    """

    start_time_ms: int
    end_time_ms: int

    scene: str
    actions: list[ObservedAction]
    actors: list[ObservedActor]
    objects: list[ObservedObject]
    uncertainties: list[str]

    @model_validator(mode="after")
    def require_action_timestamps(self) -> "VideoSegment":
        """Require timestamps for every action in a video segment."""

        for action in self.actions:
            if (
                action.start_time_ms is None
                or action.end_time_ms is None
            ):
                raise ValueError(
                    "Video actions must contain start and end timestamps"
                )

        return self

    @model_validator(mode="after")
    def number_repeated_entities(self) -> "VideoSegment":
        """Number repeated object and actor names in their observation order."""

        self._number_repeated_names(self.objects)
        self._number_repeated_names(self.actors)
        return self

    @staticmethod
    def _number_repeated_names(
        entities: list[ObservedObject] | list[ObservedActor],
    ) -> None:
        """Append one-based indices to every occurrence of a repeated name."""

        name_counts = Counter(entity.name for entity in entities)
        next_number: Counter[str] = Counter()

        for entity in entities:
            original_name = entity.name
            if name_counts[original_name] > 1:
                next_number[original_name] += 1
                entity.name = f"{original_name}_{next_number[original_name]}"


@dataclass(frozen=True)
class InstructionReferenceIndex:
    """Read-only instruction-wide identity and reverse-reference indexes."""

    segments_by_id: Mapping[SegmentId, Segment]
    actions_by_id: Mapping[ActionId, ObservedAction]
    actors_by_id: Mapping[ActorId, ObservedActor]
    objects_by_id: Mapping[ObjectId, ObservedObject]
    actor_action_ids: Mapping[ActorId, tuple[ActionId, ...]]
    object_action_ids: Mapping[ObjectId, tuple[ActionId, ...]]


class Instruction:
    """A collection of segments with observations indexed by segment ID.

    The instruction owns a validated, read-only reference index. Every action
    endpoint must resolve to an actor or object declared in the same segment.
    """

    def __init__(
        self,
        id: str,
        name: str,
        segments: list[Segment],
    ):
        """Initialize an instruction and build its per-segment lookup maps."""

        self.id = id
        self.name = name
        self.segments = segments
        self.reference_index = self._build_reference_index(segments)
        self.objects = MappingProxyType(
            {segment.id: tuple(segment.objects) for segment in segments}
        )
        self.actors = MappingProxyType(
            {segment.id: tuple(segment.actors) for segment in segments}
        )

    @staticmethod
    def _duplicates(values: list[str]) -> list[str]:
        return sorted(value for value, count in Counter(values).items() if count > 1)

    @classmethod
    def _build_reference_index(
        cls,
        segments: list[Segment],
    ) -> InstructionReferenceIndex:
        """Validate identities and endpoints, then build all reference maps."""

        duplicate_ids = {
            "segment": cls._duplicates([segment.id for segment in segments]),
            "action": cls._duplicates(
                [action.id for segment in segments for action in segment.actions]
            ),
            "actor": cls._duplicates(
                [actor.id for segment in segments for actor in segment.actors]
            ),
            "object": cls._duplicates(
                [
                    observed_object.id
                    for segment in segments
                    for observed_object in segment.objects
                ]
            ),
        }
        for kind, duplicates in duplicate_ids.items():
            if duplicates:
                scope = "globally " if kind != "segment" else ""
                raise ValueError(
                    f"Instruction {kind} IDs must be {scope}unique; "
                    f"duplicates: {duplicates!r}"
                )

        segments_by_id = {segment.id: segment for segment in segments}
        actions_by_id: dict[ActionId, ObservedAction] = {}
        actors_by_id: dict[ActorId, ObservedActor] = {}
        objects_by_id: dict[ObjectId, ObservedObject] = {}
        actor_action_ids: dict[ActorId, list[ActionId]] = {}
        object_action_ids: dict[ObjectId, list[ActionId]] = {}

        for segment in segments:
            segment_actor_ids = {actor.id for actor in segment.actors}
            segment_object_ids = {
                observed_object.id for observed_object in segment.objects
            }
            actors_by_id.update((actor.id, actor) for actor in segment.actors)
            objects_by_id.update(
                (observed_object.id, observed_object)
                for observed_object in segment.objects
            )
            actor_action_ids.update((actor.id, []) for actor in segment.actors)
            object_action_ids.update(
                (observed_object.id, []) for observed_object in segment.objects
            )

            for action in segment.actions:
                if action.actor_id not in segment_actor_ids:
                    raise ValueError(
                        f"Action {action.id!r} references unknown actor "
                        f"{action.actor_id!r} in segment {segment.id!r}"
                    )
                object_refs = (
                    action.object_id,
                    action.instrument_id,
                    action.target_id,
                )
                unknown_object_ids = sorted(
                    {
                        object_id
                        for object_id in object_refs
                        if object_id is not None and object_id not in segment_object_ids
                    }
                )
                if unknown_object_ids:
                    raise ValueError(
                        f"Action {action.id!r} references unknown objects "
                        f"{unknown_object_ids!r} in segment {segment.id!r}"
                    )

                actions_by_id[action.id] = action
                actor_action_ids[action.actor_id].append(action.id)
                for object_id in dict.fromkeys(object_refs):
                    if object_id is not None:
                        object_action_ids[object_id].append(action.id)

        for actor_id, actor in actors_by_id.items():
            actor.action_ids = list(actor_action_ids.get(actor_id, ()))
        for object_id, observed_object in objects_by_id.items():
            observed_object.action_ids = list(object_action_ids.get(object_id, ()))

        return InstructionReferenceIndex(
            segments_by_id=MappingProxyType(segments_by_id),
            actions_by_id=MappingProxyType(actions_by_id),
            actors_by_id=MappingProxyType(actors_by_id),
            objects_by_id=MappingProxyType(objects_by_id),
            actor_action_ids=MappingProxyType(
                {key: tuple(value) for key, value in actor_action_ids.items()}
            ),
            object_action_ids=MappingProxyType(
                {key: tuple(value) for key, value in object_action_ids.items()}
            ),
        )

    def extract_actors(self) -> list[ObservedActor]:
        """Extract a de-duplicated list of actors across all segments."""

        actors_by_name: dict[str, ObservedActor] = {}
        for segment_actors in self.actors.values():
            for actor in segment_actors:
                existing = actors_by_name.get(actor.name)
                if existing is None:
                    actors_by_name[actor.name] = actor.model_copy(deep=True)
                else:
                    existing.merge(actor)
        return list(actors_by_name.values())
