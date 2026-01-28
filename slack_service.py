# slack_service.py
import hashlib
import threading
from collections import Counter
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

from slack_helper import SlackFetcher
from slack_predictor import fetch_and_predict_slack  # refactored function

MIN_CLUSTER_SIZE = 2
DISTANCE_THRESHOLD = 0.25
DATA_COLUMNS = [
    "person_name",
    "date",
    "time_24h",
    "question",
    "Predicted_Topic_ID",
    "Predicted_Topic",
    "Subtopic_ID",
    "Subtopic_Code",
    "Subtopic_Size",
]


# ---------------- LIVE TRACKER ----------------
class LiveTopicTracker:
    """Maintain live Slack message clustering state and listener."""

    def __init__(self, embedder: SentenceTransformer, centroid_embeddings: np.ndarray, centroid_topics: List[str], topic_to_id: Dict[str, int]):
        self.embedder = embedder
        self.centroid_embeddings = centroid_embeddings
        self.centroid_topics = centroid_topics
        self.topic_to_id = topic_to_id

        self.lock = threading.Lock()
        self.rows: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        self.topic_buffers: Dict[str, List[int]] = {}
        self.topic_id_map: Dict[str, int] = topic_to_id.copy()
        self.processed_ids: Set[str] = set()

        self.fetcher = SlackFetcher()
        self.listener_thread: Optional[threading.Thread] = None
        self.listener_started = False

    def _predict_topic(self, emb: np.ndarray) -> str:
        sims = cosine_similarity(emb.reshape(1, -1), self.centroid_embeddings)[0]
        return self.centroid_topics[int(np.argmax(sims))]

    def _get_topic_id(self, topic: str) -> int:
        if topic not in self.topic_id_map:
            self.topic_id_map[topic] = len(self.topic_id_map)
        return self.topic_id_map[topic]

    def _build_message_id(self, message: Dict[str, str]) -> str:
        parts = [
            str(message.get("date", "")),
            str(message.get("time", "")),
            str(message.get("user_name", "")),
            str(message.get("message", ""))
        ]
        raw = "\u0001".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def handle_message(self, message: Dict[str, str]) -> None:
        text = str(message.get("message", "")).strip()
        if not text:
            return

        msg_id = self._build_message_id(message)

        with self.lock:
            if msg_id in self.processed_ids:
                return
            self.processed_ids.add(msg_id)

        emb = self.embedder.encode([text], convert_to_numpy=True)
        emb = normalize(emb)[0]

        topic = self._predict_topic(emb)
        topic_id = self._get_topic_id(topic)

        with self.lock:
            row_idx = len(self.rows)
            self.embeddings.append(emb)
            self.rows.append(
                {
                    "person_name": message.get("user_name"),
                    "date": message.get("date"),
                    "time_24h": message.get("time"),
                    "question": text,
                    "Predicted_Topic_ID": topic_id,
                    "Predicted_Topic": topic,
                    "Subtopic_ID": "Sync",
                    "Subtopic_Code": "Sync",
                    "Subtopic_Size": "Sync",
                }
            )
            self.topic_buffers.setdefault(topic, []).append(row_idx)

    def start_listener(self) -> None:
        if self.listener_started:
            return
        self.listener_thread = threading.Thread(
            target=self.fetcher.start_live_listener,
            args=(self.handle_message,),
            daemon=True,
            name="SlackLiveListener",
        )
        self.listener_thread.start()
        self.listener_started = True

    def dataframe(self) -> pd.DataFrame:
        with self.lock:
            if not self.rows:
                return pd.DataFrame(columns=DATA_COLUMNS)
            
            # Create dataframe and ensure Subtopic_ID is string
            df = pd.DataFrame(self.rows, columns=DATA_COLUMNS)
            if 'Subtopic_ID' in df.columns:
                df['Subtopic_ID'] = df['Subtopic_ID'].astype(str)
            if 'Subtopic_Code' in df.columns:
                df['Subtopic_Code'] = df['Subtopic_Code'].astype(str)
            if 'Subtopic_Size' in df.columns:
                df['Subtopic_Size'] = df['Subtopic_Size'].astype(str)
                
            return df

    def pop_live_rows(self):
        if not hasattr(self, "rows") or not self.rows:
            return []

        data = self.rows.copy()
        self.rows.clear()
        return data


# ---------------- HELPERS ----------------
def get_history_df() -> pd.DataFrame:
    """Fetch Slack history and predict topics/subtopics."""
    return fetch_and_predict_slack()


def get_live_tracker():
    """Load embedder and centroid info for live tracking."""
    import pickle
    MODEL_PATH = "/Users/rishita/Downloads/hcl_key/classification/topic_subtopic_model.pkl"
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    embedder = SentenceTransformer(model_data["embed_model"])
    centroid_embeddings = model_data["centroid_embeddings"]
    centroid_topics = model_data["centroid_topics"]
    topic_to_id = model_data.get("topic_to_id")
    return LiveTopicTracker(embedder, centroid_embeddings, centroid_topics, topic_to_id)
