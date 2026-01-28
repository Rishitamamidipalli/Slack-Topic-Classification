import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

import pickle
import pandas as pd
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------------- CONFIG ----------------
INPUT_CSV = "/Users/rishita/Downloads/hcl_key/classification/csv2.csv"
OUTPUT_CSV = "/Users/rishita/Downloads/hcl_key/classification/output_topic_subtopic.csv"
MODEL_PATH = "/Users/rishita/Downloads/hcl_key/classification/topic_subtopic_model.pkl"
EXTERNAL_CSV = "/Users/rishita/Downloads/hcl_key/classification/dataset.csv"

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
TEST_SIZE = 0.2
RANDOM_STATE = 42
DISTANCE_THRESHOLD = 0.35
MIN_CLUSTER_SIZE = 2
SUBTOPIC_SIM_THRESHOLD = 0.7
TOPIC_SIM_THRESHOLD = 0.45
UNKNOWN_TOPIC = "Unknown"
UNKNOWN_TOPIC_ID = -1

# ---------------- Topic Descriptions ----------------
topic_descriptions = {
    "Access Request Basics": (
    "How to get access, request access, gain access, or obtain permission for any system, "
    "application, software, platform, tool, or service including ETL tools, data tools, "
    "Oracle tools, databases, reporting tools, internal applications, portals, and systems. "
    "Includes questions about access approval, access provisioning, role assignment, "
    "permissions, login enablement, onboarding access for new joiners, mirroring access "
    "from one user to another, and all general access-related requests."
),

    "Contractors and New Hires": (
        "Onboarding procedures for contractors, consultants, and new employees, "
        "including initial system access, role assignment, badge and account creation, "
        "joining formalities, approvals, and day-one access issues."
    ),

    "Data Access Sets": (
        "Managing data access sets, business units, ledgers, responsibilities, "
        "security profiles, permission grouping, data visibility, "
        "and assigning or modifying data-level access."
    ),

    "Data Quality and Correction": (
        "Identifying, validating, correcting, and reconciling inaccurate or missing data, "
        "data mismatches, incorrect values, data cleanup, and master data corrections."
    ),

    "Error Resolution": (
        "Troubleshooting and resolving system, application, and processing errors, "
        "error messages, failures, root cause analysis, and corrective actions."
    ),

    "Escalation and Support": (
        "Support processes, ticket escalation procedures, severity handling, "
        "service desk interaction, SLA breaches, and issue follow-up."
    ),

    "Executives and Delegation": (
        "Executive user access, approval delegation, proxy approvals, "
        "temporary or permanent delegation of authority, and executive role handling."
    ),

    "Invoice Processing": (
        "Invoice submission, validation, approval workflows, invoice matching, "
        "exceptions, holds, rejections, and payment-related invoice issues."
    ),

    "Journals and ADFDI": (
        "Journal entry creation, upload, validation, and processing using ADFDI, "
        "spreadsheet uploads, corrections, and journal posting issues."
    ),

    "Navigation and Links": (
        "System navigation, menu paths, screen access, missing links, "
        "UI navigation help, and locating specific pages or functions."
    ),

    "Notifications": (
        "System notifications, alerts, approval emails, workflow messages, "
        "missing or delayed notifications, and communication settings."
    ),

    "Payment and Settlement": (
        "Payments, settlements, disbursements, reconciliations, "
        "failed payments, bank processing, and payment status inquiries."
    ),

    "Privileged and Exam Roles": (
        "Privileged access requests, emergency access, exam roles, "
        "temporary elevated permissions, and audit-controlled roles."
    ),

    "Purchase Requisitions": (
        "Creating, submitting, modifying, and approving purchase requisitions, "
        "PR workflows, approvals, rejections, and requisition lifecycle questions."
    ),

    "Reactivation and Deactivation": (
        "User access reactivation, deactivation, termination handling, "
        "account disabling, re-enabling access, and offboarding processes."
    ),

    "Reporting and Analytics": (
        "Reports, dashboards, analytics, data extraction, "
        "BI tools, scheduled reports, and reporting access issues."
    ),

    "Role Selection": (
        "Choosing appropriate user roles, role mapping, role conflicts, "
        "role approvals, and ensuring correct access based on job function."
    ),

    "Security and Controls": (
        "Security policies, access controls, audits, compliance requirements, "
        "segregation of duties, risk management, and control enforcement."
    ),

    "System Status and Performance": (
        "System availability, downtime, slowness, performance degradation, "
        "outages, maintenance windows, and system health issues."
    ),

    "Timeframes and Sync": (
        "Batch jobs, processing schedules, synchronization delays, "
        "data refresh timings, and background job execution."
    ),

    "Training and Guides": (
        "Training materials, user guides, SOPs, documentation, "
        "learning resources, and how-to instructions."
    ),

    "Vendor and Supplier Management": (
        "Vendor and supplier setup, maintenance, access, onboarding, "
        "updates, approvals, and supplier-related processes."
    ),

    "Workflow and Routing": (
        "Workflow configuration, routing rules, approval chains, "
        "task assignment, process automation, and workflow troubleshooting."
    )
}


# ---------------- Load Data ----------------
print("Loading dataset...")
df = pd.read_csv(INPUT_CSV, on_bad_lines="skip")
df["question"] = df["question"].astype(str)
df["topic"] = df["topic"].astype(str)

# ---------------- Train/Test Split ----------------
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["topic"]
)

# ---------------- Load Embedder ----------------
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

# ---------------- Encoding ----------------
def encode_with_desc(questions, topics):
    texts = [
        f"{q} [Topic: {topic_descriptions.get(t, t)}]"
        for q, t in zip(questions, topics)
    ]
    emb = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return normalize(emb)

print("Generating embeddings...")
train_embeddings = encode_with_desc(train_df["question"].tolist(), train_df["topic"].tolist())
test_embeddings = encode_with_desc(test_df["question"].tolist(), test_df["topic"].tolist())

train_df["embedding"] = list(train_embeddings)
test_df["embedding"] = list(test_embeddings)

# ---------------- Build Topic Centroids ----------------
def build_topic_centroids(embeddings, topics):
    topic_map = {}
    for emb, t in zip(embeddings, topics):
        topic_map.setdefault(t, []).append(emb)

    names, cents = [], []
    for t, embs in topic_map.items():
        c = np.mean(embs, axis=0)
        c /= np.linalg.norm(c)
        names.append(t)
        cents.append(c)
    return np.array(cents), names

print("Building topic centroids...")
centroid_embeddings, centroid_topics = build_topic_centroids(train_embeddings, train_df["topic"].tolist())

# Fixed topic ID mapping
topic_to_id = {t: i for i, t in enumerate(centroid_topics)}
topic_to_id[UNKNOWN_TOPIC] = UNKNOWN_TOPIC_ID

# ---------------- Predict Topic ----------------
def predict_topic(emb):
    sims = cosine_similarity(emb.reshape(1, -1), centroid_embeddings)[0]
    max_sim = sims.max()
    if max_sim < TOPIC_SIM_THRESHOLD:
        return UNKNOWN_TOPIC
    return centroid_topics[sims.argmax()]

train_df["Predicted_Topic"] = [predict_topic(e) for e in train_df["embedding"]]
train_df["Predicted_Topic_ID"] = train_df["Predicted_Topic"].map(topic_to_id)

test_df["Predicted_Topic"] = [predict_topic(e) for e in test_df["embedding"]]
test_df["Predicted_Topic_ID"] = test_df["Predicted_Topic"].map(topic_to_id)

# ---------------- SUB-CLUSTERING (TRAINING ONLY) ----------------
print("Sub-clustering within topics...")
subtopic_centroids = {}
train_rows = []

for topic, topic_id in topic_to_id.items():
    if topic == UNKNOWN_TOPIC:
        continue

    topic_df = train_df[train_df["Predicted_Topic"] == topic].reset_index(drop=True)
    topic_embs = np.vstack(topic_df["embedding"].values)

    if len(topic_df) < MIN_CLUSTER_SIZE:
        labels = np.arange(len(topic_df))
    else:
        dist_matrix = cosine_distances(topic_embs)
        upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        eps = max(0.05, np.percentile(upper, 20))
        dbscan = DBSCAN(
            eps=eps,
            min_samples=MIN_CLUSTER_SIZE,
            metric="cosine"
        )
        labels = dbscan.fit_predict(topic_embs)

    topic_df["Subtopic_ID"] = labels
    cluster_sizes = Counter(labels)

    subtopic_code_map = {}
    next_id = 0
    for cid, size in cluster_sizes.items():
        if cid == -1 or size < MIN_CLUSTER_SIZE:
            subtopic_code_map[cid] = f"{topic_id}_noise"
        else:
            subtopic_code_map[cid] = f"{topic_id}_{next_id}"
            next_id += 1

    topic_df["Subtopic_Code"] = topic_df["Subtopic_ID"].map(subtopic_code_map)
    topic_df["Subtopic_Size"] = topic_df["Subtopic_ID"].map(cluster_sizes)

    # Store centroids for test/external assignment
    valid = topic_df[~topic_df["Subtopic_Code"].str.endswith("_noise")]
    if len(valid) > 0:
        sub_map = {}
        for code, g in valid.groupby("Subtopic_Code"):
            c = np.mean(np.vstack(g["embedding"].values), axis=0)
            c /= np.linalg.norm(c)
            sub_map[code] = (int(code.split("_")[1]), c)
        subtopic_centroids[topic] = sub_map

    train_rows.append(topic_df)

train_df = pd.concat(train_rows, ignore_index=True)

# ---------------- Function to assign subtopics ----------------
def assign_subtopic_info(topic, emb):
    if topic == UNKNOWN_TOPIC or topic not in subtopic_centroids:
        return -1, f"{topic_to_id.get(topic, UNKNOWN_TOPIC_ID)}_noise", 0
    codes, vals = zip(*subtopic_centroids[topic].items())
    sub_ids, cents = zip(*vals)
    sims = cosine_similarity(emb.reshape(1, -1), np.vstack(cents))[0]
    best_idx = sims.argmax()
    if sims[best_idx] < SUBTOPIC_SIM_THRESHOLD:
        return -1, f"{topic_to_id[topic]}_noise", 0
    code = codes[best_idx]
    return sub_ids[best_idx], code, sum([1 for c in train_df["Subtopic_Code"] if c == code])

# ---------------- Assign subtopics for test ----------------
test_sub_info = [assign_subtopic_info(t, e) for t, e in zip(test_df["Predicted_Topic"], test_df["embedding"])]
test_df["Subtopic_ID"], test_df["Subtopic_Code"], test_df["Subtopic_Size"] = zip(*test_sub_info)

# ---------------- Predict on External CSV ----------------
print("Predicting on external CSV...")
df_ext = pd.read_csv(EXTERNAL_CSV, on_bad_lines="skip")
df_ext["question"] = df_ext["question"].astype(str)
df_ext["topic"] = df_ext["topic"].astype(str)
ext_embeddings = encode_with_desc(df_ext["question"].tolist(), df_ext["topic"].tolist())
df_ext["embedding"] = list(ext_embeddings)
df_ext["Predicted_Topic"] = [predict_topic(e) for e in df_ext["embedding"]]
df_ext["Predicted_Topic_ID"] = df_ext["Predicted_Topic"].map(topic_to_id)

# Assign subtopics for external data
ext_sub_info = [assign_subtopic_info(t, e) for t, e in zip(df_ext["Predicted_Topic"], df_ext["embedding"])]
df_ext["Subtopic_ID"], df_ext["Subtopic_Code"], df_ext["Subtopic_Size"] = zip(*ext_sub_info)

# ---------------- Confusion Matrix & Accuracy ----------------
# ---------------- Confusion Matrix & Accuracy ----------------
def evaluate(df, dataset_name="Dataset"):
    y_true = df["topic"].map(topic_to_id).fillna(UNKNOWN_TOPIC_ID).astype(int)
    y_pred = df["Predicted_Topic_ID"].astype(int)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    unknown_count = sum(df["Predicted_Topic"] == UNKNOWN_TOPIC)
    print(f"--- Evaluation: {dataset_name} ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc:.2f}")
    print(f"Number of Unknown predictions: {unknown_count}\n")

print("Evaluation on test split:")
evaluate(test_df, "Test Split")

print("Evaluation on external CSV:")
evaluate(df_ext, "External CSV")

# ---------------- Save CSV ----------------
output_df = pd.concat([train_df, test_df, df_ext], ignore_index=True)
output_cols = [
    "person_name", "date", "time_24h", "question",
    "Predicted_Topic_ID", "Predicted_Topic",
    "Subtopic_ID", "Subtopic_Code", "Subtopic_Size"
]
output_df[output_cols].to_csv(OUTPUT_CSV, index=False)
print(f"✅ Output CSV saved at: {OUTPUT_CSV}")

# ---------------- Save Model ----------------
print("Saving model...")
with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "embed_model": EMBED_MODEL,
        "centroid_embeddings": centroid_embeddings,
        "centroid_topics": centroid_topics,
        "topic_to_id": topic_to_id,
        "topic_descriptions": topic_descriptions,
    }, f)
print(f"✅ Model saved at: {MODEL_PATH}")
print("✅ PROCESS COMPLETE")
