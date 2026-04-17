from typing import Any, Iterator, Optional
from google.cloud import firestore
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, RunnableConfig, CheckpointTuple
import json

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
