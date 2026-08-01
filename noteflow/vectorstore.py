"""Optional vector store, using Qdrant instead of SQLite blobs.

SQLite is the default and is fine at this size: comparing a few hundred
vectors with numpy takes under a millisecond. This file is the path for when
the note count grows, because Qdrant keeps an index instead of scanning
every row.

It runs from a local folder here, so there is no server to install. The same
code works against a real Qdrant server by changing the client line.

Switched on with NOTEFLOW_VECTORDB=1.
"""

import os

STORE_PATH = os.getenv("NOTEFLOW_VECTOR_PATH", "qdrant_data")
COLLECTION = "notes"
VECTOR_SIZE = 384          # multilingual-e5-small

client = None


def enabled():
    """True if we should use Qdrant instead of SQLite blobs."""
    return os.getenv("NOTEFLOW_VECTORDB", "") == "1"


def get_client():
    """Open the store once, the first time we need it."""
    global client

    if client is None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(path=STORE_PATH)

        names = [c.name for c in client.get_collections().collections]
        if COLLECTION not in names:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE,
                                            distance=Distance.COSINE),
            )

    return client


def save(note_id, chunks, vectors):
    """Store the chunks of one note, replacing whatever was there before."""
    from qdrant_client.models import PointStruct

    store = get_client()
    remove(note_id)

    points = []
    for i, (text, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                # A note can have several chunks, so the id has to be unique.
                id=note_id * 1000 + i,
                vector=vector.tolist(),
                payload={"note_id": note_id, "text": text},
            )
        )

    store.upsert(collection_name=COLLECTION, points=points)


def remove(note_id):
    """Delete every chunk belonging to one note."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    get_client().delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="note_id", match=MatchValue(value=note_id))]
        ),
    )


def search(query_vector, limit=20):
    """Find the closest chunks and return {note_id: best score}.

    A note can have several chunks, so we keep the best score for each note.
    """
    found = get_client().query_points(
        collection_name=COLLECTION,
        query=query_vector.tolist(),
        limit=limit,
    ).points

    scores = {}
    for point in found:
        note_id = point.payload["note_id"]
        if point.score > scores.get(note_id, 0.0):
            scores[note_id] = point.score

    return scores


def count():
    """How many chunks are stored."""
    return get_client().count(collection_name=COLLECTION).count
