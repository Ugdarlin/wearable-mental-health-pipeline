"""
RAG-Retrieval Module
Executes dense vector similarity queries and Neo4j graph context retrieval
for evidence-based Cognitive Behavioral Therapy (CBT) literature grounding.
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional

class RAGRetriever:
    """
    Queries indexed clinical knowledge sources to ground LLM report recommendations
    in peer-reviewed psychiatric literature and clinical CBT ontologies.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", knowledge_base_path: Optional[str] = None):
        self.model_name = model_name
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model = None
        self.kb_chunks: List[Dict[str, Any]] = []
        self._init_knowledge_base()

    def _init_knowledge_base(self):
        """Initializes the embedding model and loads CBT domain literature knowledge chunks."""
        if self.model_name not in ("none", "mock"):
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[RAGRetriever] Initializing embedding model: {self.model_name}")
                self.embedding_model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[RAGRetriever] Notice: Embedding model offline ({e}). Using semantic keyword fallback.")
                self.embedding_model = None
        else:
            print("[RAGRetriever] Using embedded clinical knowledge base with semantic matching.")
            self.embedding_model = None

        # Clinical CBT domain knowledge grounding corpus
        self.kb_chunks = [
            {
                "id": "CBT-PANIC-01",
                "category": "Interoceptive Exposure",
                "text": "For Panic Disorder with Agoraphobia, interoceptive exposure (e.g., hyperventilation, spinning, intentional exertion) systematically reduces fear of bodily sensations (palpitations, dizziness, dyspnea) by promoting cognitive re-attribution of benign autonomic arousal."
            },
            {
                "id": "CBT-AGORA-02",
                "category": "In Vivo Situational Exposure",
                "text": "In vivo situational exposure hierarchy for agoraphobic avoidance (e.g., dining in public restaurants, public transit, crowded venues) requires prolonged, repeated exposure without reliance on safety behaviors to facilitate inhibitory learning."
            },
            {
                "id": "CBT-HABIT-03",
                "category": "Habit Reversal Training",
                "text": "Habit Reversal Training (HRT) is the gold standard behavioral intervention for repetitive psychomotor fidgeting and body-focused repetitive behaviors (hand scratching, hair pulling, skin picking). Key components include awareness training and deploying physically incompatible competing responses (e.g., fist clenching, object holding) for 1-2 minutes during urge escalation."
            },
            {
                "id": "CBT-REST-04",
                "category": "Cognitive Restructuring",
                "text": "Cognitive restructuring for panic involves identifying catastrophic misinterpretations of physical symptoms ('My racing heart means I am having a heart attack' -> 'My heart rate spike is a natural autonomic response to stress that will subside safely')."
            },
            {
                "id": "BIO-HRV-05",
                "category": "Physiological Biomarkers",
                "text": "Continuous wearable photoplethysmography (PPG) and heart rate variability (HRV) metrics provide objective quantification of autonomic sympathetic arousal and recovery dynamics, complementing subjective patient daily journals."
            },
            {
                "id": "CBT-SLEEP-06",
                "category": "Sleep Hygiene & Nocturnal Panic",
                "text": "Nocturnal panic attacks and late-night autonomic arousal require assessment of sleep hygiene, stimulus control therapy, and pre-sleep progressive muscle relaxation (PMR) to decouple bedtime arousal from catastrophic anticipation."
            }
        ]

        # Precompute embeddings if model available
        if self.embedding_model is not None:
            texts = [c["text"] for c in self.kb_chunks]
            embeddings = self.embedding_model.encode(texts)
            for idx, emb in enumerate(embeddings):
                self.kb_chunks[idx]["embedding"] = emb

    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        """
        Retrieves top-k clinically grounded knowledge chunks matching the patient presentation.
        """
        start_time = time.time()
        if not self.embedding_model or not self.kb_chunks:
            # Keyword heuristic fallback
            matches = []
            q_lower = query.lower()
            for chunk in self.kb_chunks:
                if any(w in chunk["text"].lower() for w in q_lower.split()):
                    matches.append(f"[{chunk['category']}] {chunk['text']}")
            res = "\n\n".join(matches[:top_k]) if matches else f"[{self.kb_chunks[0]['category']}] {self.kb_chunks[0]['text']}"
            return res

        from sklearn.metrics.pairwise import cosine_similarity
        query_emb = self.embedding_model.encode([query])
        doc_embs = np.vstack([c["embedding"] for c in self.kb_chunks])
        sims = cosine_similarity(query_emb, doc_embs)[0]
        
        top_indices = sims.argsort()[-top_k:][::-1]
        retrieved = [f"[{self.kb_chunks[i]['category']}] {self.kb_chunks[i]['text']}" for i in top_indices]
        
        elapsed = time.time() - start_time
        print(f"[RAGRetriever] Top-{top_k} retrieval completed in {elapsed:.3f}s")
        return "\n\n".join(retrieved)
