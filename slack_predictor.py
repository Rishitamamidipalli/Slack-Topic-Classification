import pickle
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import CountVectorizer

from sentence_transformers import SentenceTransformer

from slack_helper import SlackFetcher

# ---------------- CONFIG ----------------
MODEL_PATH = "/Users/rishita/Downloads/hcl_key/classification/topic_subtopic_model.pkl"

MIN_CLUSTER_SIZE = 2
DISTANCE_THRESHOLD = 0.35

TOPIC_SIM_THRESHOLD = 0.30
UNKNOWN_TOPIC = "Unknown"
UNKNOWN_TOPIC_ID = -1
ACCESS_KEYWORDS = [
    "access", "permission", "grant", "role",
    "etl", "tool", "tools",
    "oracle", "database", "db",
    "login", "onboarding", "new joiner",
    "mirror", "mirroring"
]
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
# ---------------- KEYWORD OVERRIDE ----------------
def apply_keyword_override(message):
    msg_lower = str(message).lower()
    for kw in ACCESS_KEYWORDS:
        if kw in msg_lower:
            return "Access Request Basics"
    return UNKNOWN_TOPIC
# ---------------- SAFE KEYWORD EXTRACTION ----------------
def extract_top_keywords(texts, top_k=3):
    # Filter out too-short or non-string messages
    cleaned = [t for t in texts if isinstance(t, str) and len(t.split()) >= 2]

    if not cleaned:
        return UNKNOWN_TOPIC

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50)

    try:
        X = vectorizer.fit_transform(cleaned)
    except ValueError:
        return UNKNOWN_TOPIC

    if X.shape[1] == 0:
        return UNKNOWN_TOPIC

    freqs = X.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    top_idx = freqs.argsort()[::-1][:top_k]

    return ", ".join(words[i] for i in top_idx)

# ---------------- MAIN FUNCTION ----------------
def fetch_and_predict_slack() -> pd.DataFrame:

    # ---------------- FETCH SLACK DATA ----------------
    fetcher = SlackFetcher()
    df = fetcher.fetch_history()
    if df.empty:
        return pd.DataFrame(columns=DATA_COLUMNS)

    # ---------------- LOAD MODEL ----------------
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    embedder = SentenceTransformer(model_data["embed_model"])
    centroid_embeddings = normalize(model_data["centroid_embeddings"])
    centroid_topics = model_data["centroid_topics"]

    topic_to_id = model_data.get(
        "topic_to_id",
        {t: i for i, t in enumerate(centroid_topics)}
    )
    topic_to_id[UNKNOWN_TOPIC] = UNKNOWN_TOPIC_ID

    # ---------------- EMBEDDINGS ----------------
    texts = df["message"].astype(str).tolist()
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    embeddings = normalize(embeddings)
    df["embedding"] = list(embeddings)

    # ---------------- TOPIC PREDICTION ----------------
    def predict_topic(emb):
        sims = cosine_similarity(emb.reshape(1, -1), centroid_embeddings)[0]
        if sims.max() < TOPIC_SIM_THRESHOLD:
            return UNKNOWN_TOPIC
        return centroid_topics[sims.argmax()]

    df["Predicted_Topic"] = df["embedding"].apply(predict_topic)
    df["Predicted_Topic_ID"] = df["Predicted_Topic"].map(topic_to_id)

    output_rows = []

    # ---------------- APPLY KEYWORD OVERRIDE ----------------
    unknown_mask = df["Predicted_Topic"] == UNKNOWN_TOPIC
    df.loc[unknown_mask, "Predicted_Topic"] = df.loc[unknown_mask, "message"].apply(apply_keyword_override)
    df["Predicted_Topic_ID"] = df["Predicted_Topic"].map(topic_to_id)


    # ==================================================
    # 1️⃣ HANDLE UNKNOWN TOPICS (DBSCAN)
    # ==================================================
    unknown_df = df[df["Predicted_Topic"] == UNKNOWN_TOPIC].reset_index(drop=True)

    if not unknown_df.empty:
        unknown_embs = np.vstack(unknown_df["embedding"].values)

        dbscan = DBSCAN(eps=0.4, min_samples=2, metric="cosine")
        labels = dbscan.fit_predict(unknown_embs)
        unknown_df["cluster"] = labels

        for cid in sorted(unknown_df["cluster"].unique()):
            cluster_df = unknown_df[unknown_df["cluster"] == cid]

            if cid == -1:
                topic_name = UNKNOWN_TOPIC
            else:
                topic_name = extract_top_keywords(cluster_df["message"].tolist())

            for _, r in cluster_df.iterrows():
                output_rows.append({
                    "person_name": r.get("user_name"),
                    "date": r.get("date"),
                    "time_24h": r.get("time"),
                    "question": r["message"],
                    "Predicted_Topic_ID": UNKNOWN_TOPIC_ID,
                    "Predicted_Topic": topic_name,
                    "Subtopic_ID": -1,
                    "Subtopic_Code": "unknown",
                    "Subtopic_Size": "unknown"
                })

    # ==================================================
    # 2️⃣ HANDLE KNOWN TOPICS (AGGLOMERATIVE)
    # ==================================================
    for topic, topic_id in topic_to_id.items():
        if topic == UNKNOWN_TOPIC:
            continue

        topic_df = df[df["Predicted_Topic"] == topic].reset_index(drop=True)
        if topic_df.empty:
            continue

        topic_embs = np.vstack(topic_df["embedding"].values)

        if len(topic_df) < MIN_CLUSTER_SIZE:
            labels = np.arange(len(topic_df))
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=DISTANCE_THRESHOLD,
                metric="cosine",
                linkage="average"
            )
            labels = clusterer.fit_predict(topic_embs)

        sizes = Counter(labels)
        subtopic_code_map = {}
        next_id = 0
        for cid, size in sizes.items():
            if size < MIN_CLUSTER_SIZE:
                subtopic_code_map[int(cid)] = f"{topic_id}_noise"
            else:
                subtopic_code_map[int(cid)] = f"{topic_id}_{next_id}"
                next_id += 1

        topic_df["Subtopic_ID"] = labels
        topic_df["Subtopic_Code"] = topic_df["Subtopic_ID"].map(subtopic_code_map)
        topic_df["Subtopic_Size"] = topic_df["Subtopic_ID"].map(sizes)

        for _, r in topic_df.iterrows():
            output_rows.append({
                "person_name": r.get("user_name"),
                "date": r.get("date"),
                "time_24h": r.get("time"),
                "question": r["message"],
                "Predicted_Topic_ID": r["Predicted_Topic_ID"],
                "Predicted_Topic": r["Predicted_Topic"],
                "Subtopic_ID": r["Subtopic_ID"],
                "Subtopic_Code": r["Subtopic_Code"],
                "Subtopic_Size": r["Subtopic_Size"]
            })

    # ---------------- FINAL OUTPUT ----------------
    output_df = pd.DataFrame(output_rows)
    output_df = output_df.reindex(columns=DATA_COLUMNS)

    return output_df
