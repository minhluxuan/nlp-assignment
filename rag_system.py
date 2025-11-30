from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import faiss
import config


class RAGSystem:
    
    def __init__(self, documents: List[str]):
        print("Initializing RAG system...")
        
        # Initialize embedding model
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            cache_folder=config.MODEL_CACHE_DIR,
            device=config.DEVICE
        )
        
        # Initialize reranker
        print(f"Loading reranker model: {config.RERANKER_MODEL}")
        self.reranker = FlagReranker(
            config.RERANKER_MODEL,
            cache_dir=config.MODEL_CACHE_DIR,
            use_fp16=True if config.DEVICE == "cuda" else False
        )
        
        # Store documents
        self.documents = documents
        
        # Build index
        print(f"Building vector index for {len(documents)} documents...")
        self.build_index()
        
        print("RAG system initialized successfully!")
    
    def build_index(self):
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            self.documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} documents")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in FAISS index
        scores, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k, len(self.documents))
        )
        
        # Return documents with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[str, float]]:
        if top_k is None:
            top_k = config.TOP_K_RERANK
        
        if not documents:
            return []
        
        # Prepare pairs for reranking
        pairs = [[query, doc] for doc in documents]
        
        # Get reranking scores
        scores = self.reranker.compute_score(pairs, normalize=True)
        
        # Handle single document case
        if not isinstance(scores, list):
            scores = [scores]
        
        # Sort by score
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:top_k]
    
    def retrieve_and_rerank(self, query: str) -> List[Tuple[str, float]]:
        retrieved_docs = self.retrieve(query, top_k=config.TOP_K_RETRIEVAL)
        if not retrieved_docs:
            return []
        
        candidate_docs = [doc for doc, _ in retrieved_docs]
        
        reranked_docs = self.rerank(query, candidate_docs, top_k=config.TOP_K_RERANK)
        
        return reranked_docs
    
    def get_context(self, query: str) -> str:
        reranked_docs = self.retrieve_and_rerank(query)
        
        if not reranked_docs:
            return ""
        
        context_parts = []
        for i, (doc, score) in enumerate(reranked_docs, 1):
            context_parts.append(f"[Thông tin {i}] (Độ liên quan: {score:.3f})\n{doc}")
        
        return "\n\n".join(context_parts)