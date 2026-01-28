import hashlib
import time
import pandas as pd
import streamlit as st
from slack_service import get_history_df, get_live_tracker

@st.cache_resource
def _get_tracker_cached():
    tracker = get_live_tracker()
    tracker.start_listener()
    return tracker

st.set_page_config(page_title="Slack Topic Monitor", layout="wide")
st.title("Slack Topic Monitor")
st.caption("Fetching Slack messages and clustering topics live...")

# Load historical messages
history_df = get_history_df()

def _compute_message_hash(row: pd.Series) -> str:
    parts = [
        str(row.get("date", "")),
        str(row.get("time_24h", "")),
        str(row.get("person_name", "")),
        str(row.get("question", "")).strip(),
    ]
    raw = "\u0001".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _deduplicate_messages(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["__message_id"] = df.apply(_compute_message_hash, axis=1)
    df = df.drop_duplicates(subset="__message_id", keep="first")
    return df.drop(columns="__message_id")

history_df = _deduplicate_messages(history_df)

# ----------------- Sidebar (Filters + Sync History) -----------------
with st.sidebar:
    st.header("Controls")
    
    # Filters
    st.subheader("Filter Messages")
    if not history_df.empty:
        date_min = pd.to_datetime(history_df["date"]).min()
        date_max = pd.to_datetime(history_df["date"]).max()
        selected_dates = st.date_input("Select Date Range", [date_min, date_max])
        selected_start_time = st.time_input("Start Time", value=pd.to_datetime("00:00:00").time())
        selected_end_time = st.time_input("End Time", value=pd.to_datetime("23:59:59").time())
        all_topics = history_df["Predicted_Topic"].unique().tolist()
        selected_topics = st.multiselect("Select Topics", options=all_topics, default=all_topics)
    else:
        today = pd.Timestamp.now().date()
        selected_dates = (today, today)
        selected_start_time = pd.to_datetime("00:00:00").time()
        selected_end_time = pd.to_datetime("23:59:59").time()
        selected_topics = []

# ----------------- Tabs -----------------
tab1, tab2 = st.tabs(["Most Recent Messages", "Clusters & Subclusters"])
live_placeholder = tab1.empty()
cluster_placeholder = tab2.empty()

# ----------------- Live Tracker -----------------
tracker = _get_tracker_cached()

def clean_dataframe_for_display(df):
    """Convert mixed-type columns to strings for Arrow compatibility."""
    if df.empty:
        return df
    df = df.copy()
    str_columns = ['Subtopic_ID', 'Subtopic_Code', 'Subtopic_Size', 'Predicted_Topic_ID']
    for col in str_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = df['date'].dt.date
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
    if 'Subtopic_Size' in df.columns:
        df['Subtopic_Size'] = pd.to_numeric(df['Subtopic_Size'], errors='coerce')
    rename_map = {
        'person_name': 'Person',
        'date': 'Date',
        'time_24h': 'Time',
        'question': 'Message',
        'Predicted_Topic': 'Topic',
        'Subtopic_Code': 'Subcluster',
        'Subtopic_Size': 'Subcluster Size',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

# ----------------- Most Recent Messages Tab -----------------
with live_placeholder.container():
    col1, col2 = st.columns([0.9, 0.1])
    with col2:
        if st.button("Refresh"):
            live_df = tracker.dataframe()
            print(live_df[["Predicted_Topic", "question", "Subtopic_ID"]])
            live_df = _deduplicate_messages(live_df)
            combined_df = pd.concat([history_df, live_df], ignore_index=True, sort=False)
            combined_df = _deduplicate_messages(combined_df)
            combined_df = combined_df.reindex(columns=tracker.rows[0].keys() if tracker.rows else history_df.columns)
            st.session_state["combined_df"] = combined_df

    # Load combined_df from session state or initialize
    combined_df = st.session_state.get("combined_df", None)
    if combined_df is None:
        live_df = tracker.dataframe()
        live_df = _deduplicate_messages(live_df)
        combined_df = pd.concat([history_df, live_df], ignore_index=True, sort=False)
        combined_df = _deduplicate_messages(combined_df)
        combined_df = combined_df.reindex(columns=tracker.rows[0].keys() if tracker.rows else history_df.columns)
        st.session_state["combined_df"] = combined_df

    # Apply filters
    if not combined_df.empty:
        combined_df["date"] = pd.to_datetime(combined_df["date"], errors='coerce')
        combined_df["time_24h"] = pd.to_datetime(combined_df["time_24h"], format="%H:%M:%S", errors='coerce').dt.time
        mask = (
            (combined_df["date"] >= pd.to_datetime(selected_dates[0])) &
            (combined_df["date"] <= pd.to_datetime(selected_dates[1])) &
            (combined_df["time_24h"] >= selected_start_time) &
            (combined_df["time_24h"] <= selected_end_time) &
            (combined_df["Predicted_Topic"].isin(selected_topics))
        )
        combined_df = combined_df[mask].copy()

    # Display table
    if combined_df.empty:
        st.info("No Slack messages for selected filters.")
    else:
        st.subheader("Most Recent Messages")
        display_df = combined_df.sort_values(["date", "time_24h"], ascending=False)
        display_df = display_df.drop(columns=["Predicted_Topic_ID", "Subtopic_ID"], errors="ignore")
        display_df = clean_dataframe_for_display(display_df)
        st.dataframe(display_df, width="stretch")

        st.subheader("Topic Distribution")
        if 'Predicted_Topic' in combined_df.columns:
            topic_counts = combined_df["Predicted_Topic"].value_counts().sort_values(ascending=False)
            st.bar_chart(topic_counts)
        else:
            st.info("No topic data available for visualization.")

# ----------------- Cluster & Subclusters Tab -----------------
with cluster_placeholder.container():
    if combined_df.empty:
        st.info("No messages for clusters.")
    else:
        cluster_list = combined_df["Predicted_Topic"].dropna().unique().tolist()
        if not cluster_list:
            st.warning("No topics available for clustering.")
        else:
            if "selected_cluster" not in st.session_state or st.session_state["selected_cluster"] not in cluster_list:
                st.session_state["selected_cluster"] = cluster_list[0]

            selected_cluster = st.selectbox(
                "Select Cluster/Topic",
                options=cluster_list,
                index=cluster_list.index(st.session_state["selected_cluster"]),
                key="live_cluster_dropdown"
            )
            st.session_state["selected_cluster"] = selected_cluster

            cluster_df = combined_df[combined_df["Predicted_Topic"] == selected_cluster]
            if not cluster_df.empty and "Subtopic_Code" in cluster_df.columns:
                for sub in cluster_df["Subtopic_Code"].unique():
                    with st.expander(f"Subcluster {sub}", expanded=False):
                        sub_df = cluster_df[cluster_df["Subtopic_Code"] == sub].sort_values(["date", "time_24h"], ascending=False)
                        sub_df = sub_df.drop(columns=["Predicted_Topic_ID", "Subtopic_ID"], errors="ignore")
                        sub_df = clean_dataframe_for_display(sub_df)
                        st.dataframe(sub_df, width="stretch")
            else:
                st.info("No subclusters for this cluster.")
