from typing import Any, Iterator, Optional
from google.cloud import firestore
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, RunnableConfig, CheckpointTuple
import json

"""
class PydanticJSONSerializer:
    "Replaces the default msgpack serde with JSON that preserves Pydantic types."

    PYDANTIC_TYPES = {
        "UserProfile": UserProfile,
        "WeeklyMealPlan": WeeklyMealPlan,
        "Meal": Meal,
        "Ingredient": Ingredient,
        "planner_node_output": planner_node_output,
        "DayConstraint": DayConstraint,
    }

    MESSAGE_TYPES = {
        "HumanMessage": HumanMessage,
        "AIMessage": AIMessage,
        "SystemMessage": SystemMessage,
        "BaseMessage": BaseMessage
    }

    def dumps_typed(self, obj: Any) -> tuple[str, str]:
        "Serialize to (type_tag, json_string)."

        return ("json", json.dumps(self._encode(obj)))

    def loads_typed(self, data: tuple[str, str]) -> Any:
        "Deserialize from (type_tag, json_string)."
        _, json_str = data
        return self._decode(json.loads(json_str))

    def _encode(self, obj: Any) -> Any:
        "Recursively encode, tagging Pydantic models with their type name."

        if isinstance(obj, (HumanMessage, AIMessage, SystemMessage, BaseMessage)):
            return {
                "__lc_message__": type(obj).__name__,
                "data": obj.model_dump() if hasattr(obj, "model_dump") else obj.dict(),
            }
        elif isinstance(obj, BaseModel):
            return {
                "__pydantic_type__": type(obj).__name__,
                "data": self._encode(obj.model_dump()),
            }
        elif isinstance(obj, deque):
            return {
                "__deque__": True,
                "data": [self._encode(i) for i in obj],
            }
        elif isinstance(obj, tuple):
            return {
                "__tuple__": True,
                "data": [self._encode(i) for i in obj],
            }
        elif isinstance(obj, dict):
            return {k: self._encode(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._encode(i) for i in obj]
        try:
            json.dumps(obj)  # test if it's natively serializable
            return obj
        except TypeError:
            raise TypeError(
                f"PydanticJSONSerializer._encode: unhandled type {type(obj).__name__}. "
                "Add it to the encoder."
            )

    def _decode(self, obj: Any) -> Any:
        from collections import deque

        if isinstance(obj, dict):
            if "__lc_message__" in obj:
                type_name = obj["__lc_message__"]
                if type_name in self.MESSAGE_TYPES:
                    return self.MESSAGE_TYPES[type_name](**self._decode(obj["data"]))
            elif "__pydantic_type__" in obj:
                type_name = obj["__pydantic_type__"]
                if type_name in self.MESSAGE_TYPES:
                    return self.MESSAGE_TYPES[type_name](**self._decode(obj["data"]))
                elif type_name in self.PYDANTIC_TYPES:
                    return self.PYDANTIC_TYPES[type_name].model_validate(
                        self._decode(obj["data"])
                    )
            elif "__deque__" in obj:
                return deque(self._decode(i) for i in obj["data"])
            elif "__tuple__" in obj:
                return tuple(self._decode(i) for i in obj["data"])
            return {k: self._decode(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._decode(i) for i in obj]
        return obj
"""

class FirestoreCheckpointer(BaseCheckpointSaver):
    def __init__(self, collection_path: str = "checkpoints"):
        super().__init__()
        self.db = firestore.Client(database="nutrition-agent-store")
        self.collection = self.db.collection(collection_path)

    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: Any) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
    
        doc_ref = self.collection.document(thread_id).collection("history").document(checkpoint_id)
        doc_ref.set({
            "checkpoint": self.serde.dumps_typed(checkpoint),
            "metadata": json.dumps(metadata),
            "thread_id": thread_id,
            "ts": checkpoint.get("ts")
        })
    
        # Return the config enriched with the checkpoint_id
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        query = self.collection.document(thread_id).collection("history")

        if checkpoint_id:
            doc = query.document(checkpoint_id).get()
        else:
            docs = query.order_by("ts", direction=firestore.Query.DESCENDING).limit(1).stream()
            doc = next(docs, None)

        if doc and doc.exists:
            data = doc.to_dict()
            checkpoint = self.serde.loads_typed(data["checkpoint"])
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": doc.id,
                    }
                },
            checkpoint=checkpoint,
            metadata=json.loads(data["metadata"]),
            parent_config=None,
            )
        return None

    def list(self, config, *, before=None, limit=None) -> Iterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        query = (
            self.collection.document(thread_id)
            .collection("history")
            .order_by("ts", direction=firestore.Query.DESCENDING)
        )

        if before:
            before_id = before["configurable"].get("checkpoint_id")
            if before_id:
                before_doc = (
                    self.collection.document(thread_id)
                    .collection("history")
                    .document(before_id)
                    .get()
                )
                if before_doc.exists:
                    query = query.start_after(before_doc)

        if limit:
            query = query.limit(limit)

        for doc in query.stream():
            data = doc.to_dict()
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": doc.id,
                    }
                },
                checkpoint=self.serde.loads_typed(data["checkpoint"]),
                metadata=data["metadata"],
                parent_config=None,
            )

    def put_writes(self, config: RunnableConfig, writes: list, task_id: str) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        doc_ref = (
            self.collection.document(thread_id)
            .collection("writes")
            .document(f"{checkpoint_id}_{task_id}")
        )
        doc_ref.set({
            "writes": self.serde.dumps_typed(writes),
            "task_id": task_id,
            "checkpoint_id": checkpoint_id,
        })