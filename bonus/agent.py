"""HybridMemoryAgent — episodic memory (vector store) + user profile (feature store).

    remember(text, user_id)  chunk -> embed -> upsert (payload carries user_id)
    recall(query, user_id)   Feast profile + recent activity
                             + hybrid (BM25 + vector, RRF k=60) over THIS user's memories
                             -> assembled context string (no LLM call)

Design notes live in bonus/ARCHITECTURE.md. Reuses app.embeddings (same
EMBEDDING_BACKEND switch as the lab) and the Feast repo applied in NB4.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient, models  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from app.embeddings import Embedder  # noqa: E402

COLLECTION = "bonus_memory"
FEAST_REPO = ROOT / "app" / "feast_repo"
PROFILE_FEATURES = [
    "user_profile_features:topic_affinity",
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]
MAX_CHUNK_WORDS = 60   # ~1 ý trọn vẹn; xem ARCHITECTURE.md, quyết định 1
RRF_K = 60


def chunk(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Gom câu liên tiếp cho tới khi chạm max_words. Không cắt giữa câu."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+|\n+", text) if s.strip()]
    chunks, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class HybridMemoryAgent:
    def __init__(self, feature_store: Any | None = None) -> None:
        self.embedder = Embedder()
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=self.embedder.dim,
                                               distance=models.Distance.COSINE),
        )
        self._next_id = 0
        # Mirror of each user's chunks for BM25. Rebuilt lazily per user: one
        # user's memory is small, and a global BM25 index would leak IDF
        # statistics across tenants.
        self._texts: dict[str, list[str]] = {}
        self._bm25: dict[str, BM25Okapi] = {}
        # Recent queries kept in-process: Feast `queries_last_hour` is a
        # materialised count, this buffer is the sub-second "what did I just ask".
        self._recent: dict[str, list[tuple[float, str]]] = {}
        self.fs = feature_store if feature_store is not None else self._load_feast()

    @staticmethod
    def _load_feast():
        if not (FEAST_REPO / "registry.db").exists():
            return None   # NB4 chưa chạy -> agent vẫn chạy, chỉ thiếu profile
        try:
            from feast import FeatureStore
            return FeatureStore(repo_path=str(FEAST_REPO))
        except Exception:  # noqa: BLE001
            return None

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        parts = chunk(text)
        vectors = list(self.embedder.embed(parts))
        now = time.time()
        self.client.upsert(COLLECTION, points=[
            models.PointStruct(id=self._next_id + i, vector=v.tolist(),
                               payload={"user_id": user_id, "text": p, "ts": now})
            for i, (p, v) in enumerate(zip(parts, vectors))
        ])
        self._next_id += len(parts)
        self._texts.setdefault(user_id, []).extend(parts)
        self._bm25.pop(user_id, None)   # invalidate; rebuilt on next recall

    # ── read path ───────────────────────────────────────────────────────
    def _profile(self, user_id: str) -> dict:
        if self.fs is None:
            return {}
        try:
            raw = self.fs.get_online_features(
                features=PROFILE_FEATURES, entity_rows=[{"user_id": user_id}]).to_dict()
            return {k: v[0] for k, v in raw.items() if k != "user_id"}
        except Exception as exc:  # noqa: BLE001
            return {"_error": str(exc)}

    def _vector_ids(self, query: str, user_id: str, depth: int) -> list[str]:
        qv = next(self.embedder.embed([query])).tolist()
        # Filter INSIDE the ANN search (NB5): post-filtering a shared index by
        # user_id is both a recall cliff and a cross-user leak waiting to happen.
        hits = self.client.query_points(
            COLLECTION, query=qv, limit=depth,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="user_id", match=models.MatchValue(value=user_id))]),
        ).points
        return [h.payload["text"] for h in hits]

    def _keyword_ids(self, query: str, user_id: str, depth: int) -> list[str]:
        texts = self._texts.get(user_id, [])
        if not texts:
            return []
        if user_id not in self._bm25:
            self._bm25[user_id] = BM25Okapi([t.lower().split() for t in texts])
        scores = self._bm25[user_id].get_scores(query.lower().split())
        ranked = sorted(range(len(texts)), key=lambda i: -scores[i])
        return [texts[i] for i in ranked[:depth] if scores[i] > 0]

    def search(self, query: str, user_id: str, k: int = 3) -> list[str]:
        """Hybrid retrieval over one user's memories, fused with RRF (rank 1-based)."""
        depth = max(k * 5, 20)
        rrf: dict[str, float] = {}
        for ranked in (self._keyword_ids(query, user_id, depth),
                       self._vector_ids(query, user_id, depth)):
            for rank, t in enumerate(ranked, start=1):
                rrf[t] = rrf.get(t, 0.0) + 1.0 / (RRF_K + rank)
        return [t for t, _ in sorted(rrf.items(), key=lambda kv: -kv[1])[:k]]

    def recall(self, query: str, user_id: str = "u_001", k: int = 3) -> str:
        """Retrieve top-K memories + user profile features -> assembled context."""
        prof = self._profile(user_id)
        memories = self.search(query, user_id, k)

        now = time.time()
        recent = [q for ts, q in self._recent.get(user_id, []) if now - ts < 3600]
        self._recent.setdefault(user_id, []).append((now, query))

        def val(key, default="?"):
            v = prof.get(key)
            return default if v is None else v

        lines = [
            f"User {user_id} likes {val('topic_affinity')}, reading at "
            f"{val('reading_speed_wpm')} wpm, language={val('preferred_language')}.",
            f"Recent activity: queries_last_hour={val('queries_last_hour', 'n/a (TTL expired)')}, "
            f"distinct_topics_24h={val('distinct_topics_24h', 'n/a')}; "
            f"this session: {recent[-3:] if recent else '[]'}",
            "Top memories:",
        ]
        lines += [f"  {i}. {m}" for i, m in enumerate(memories, 1)] or ["  (none)"]
        if "_error" in prof:
            lines.append(f"[profile unavailable: {prof['_error'][:80]}]")
        return "\n".join(lines)
