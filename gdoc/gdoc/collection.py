from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    SparseVectorParams,
    SparseIndexParams,
)
from gdoc.gdoc.models import client_
client =None
client=client_()
def create_collection(COLLECTION_NAME, DENSE_DIM):
    client = None
    client = client_()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            # --- Dense vectors (BGE-M3 embeddings) ---
            # vectors_config={
            #     "dense": VectorParams(
            #         size=DENSE_DIM,
            #         distance=Distance.COSINE,
            #         on_disk=True,           # memmap — stays on disk, OS pages hot chunks
            #     )
            # },

            # --- Sparse vectors (BM25 keyword matching) ---
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=True,       # sparse index also on disk
                    )
                )
            },

            # --- HNSW graph index on disk ---
            hnsw_config=HnswConfigDiff(
                on_disk=True,
                m=16,                       # graph connectivity (16 = good default)
                ef_construct=100,           # build quality (higher = better but slower index)
            ),

            # --- Memmap threshold ---
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=10_000,    # segments >10k vectors use memmap automatically
            ),
        )
        return {
            "message":f"Collection created: {COLLECTION_NAME}"
        }
    else:
        return {
            "message":f"Collection already exists: {COLLECTION_NAME}"
        }


create_collection("metadata_docs",768)
# create_collection("large_docs",768)
# client.delete_collection("small_docs")
# client.delete_collection("large_docs")
print(client.get_collections())

# check small_docs config
# info = client.get_collection("metadata_docs")
# print(info)