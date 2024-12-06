import streamlit as st
import pandas as pd
import os
import re
from functools import partial

# Paths
ORIGINAL_CSV = "../multilabel/data/test.csv"
UPDATED_CSV = "updated/multilabel_test.csv"

# All possible labels
ALL_LABELS = [
    "teleology","telecommunications","radio frequency","radar","mobile phones","bluetooth","WiFi","data networks","optical networks","microwave technology",
    "radio technology","mobile radio","4G","LiFi","mobile network","radio and television","satellite radio","telecommunications networks","5G","fiber-optic network",
    "cognitive radio","fixed wireless network"
]

st.set_page_config(layout="wide")
st.title("Interactive Label Editor with Automatic Saving")

@st.cache_data
def load_original_data():
    df = pd.read_csv(ORIGINAL_CSV)
    df['topics'] = df['topics'].apply(eval)
    return df

def save_updated_data(df):
    df_to_save = df.copy()
    df_to_save['topics'] = df_to_save['topics'].apply(lambda x: str(x))
    os.makedirs("updated", exist_ok=True)
    df_to_save.to_csv(UPDATED_CSV, index=False)

def highlight_text(text, labels):
    def highlight_match(match):
        return f"<mark>{match.group(0)}</mark>"
    
    highlighted = text
    sorted_labels = sorted(labels, key=len, reverse=True)
    for label in sorted_labels:
        pattern = re.compile(r"\b" + re.escape(label) + r"\b", re.IGNORECASE)
        highlighted = pattern.sub(highlight_match, highlighted)
    return highlighted

if 'df' not in st.session_state:
    original_df = load_original_data()
    # We'll keep a copy for updated_df
    st.session_state.df = original_df.copy()
    if os.path.exists(UPDATED_CSV):
        updated_df = pd.read_csv(UPDATED_CSV)
        updated_df['topics'] = updated_df['topics'].apply(eval)
        st.session_state.df = updated_df
    # If there's no updated file, we start fresh from original
    save_updated_data(st.session_state.df)

df = st.session_state.df
original_df = load_original_data()

def on_checkbox_change(row_index, label):
    current_topics = df.at[row_index, 'topics']
    checkbox_key = f"cb_{row_index}_{label}"
    new_value = st.session_state[checkbox_key]
    if new_value and label not in current_topics:
        current_topics.append(label)
    elif not new_value and label in current_topics:
        current_topics.remove(label)
    df.at[row_index, 'topics'] = current_topics
    save_updated_data(df)

for i, row in df.iterrows():
    original_topics = original_df.at[i, 'topics']
    current_topics = row['topics']
    updated = set(original_topics) != set(current_topics)
    
    container = st.container()
    if updated:
        container.markdown("""
        <div style="background-color: #ffffe0; padding: 10px; border-radius:5px;">
        """, unsafe_allow_html=True)
    else:
        container.markdown("""
        <div style="background-color: #f9f9f9; padding: 10px; border-radius:5px;">
        """, unsafe_allow_html=True)

    highlighted_text = highlight_text(row['text'], ALL_LABELS)
    
    # Add [UPDATED] label if the topics differ from original
    title_str = f"**Data Point {i}**"
    if updated:
        title_str += " [UPDATED]"
    
    container.markdown(title_str, unsafe_allow_html=True)
    container.markdown(highlighted_text, unsafe_allow_html=True)
    
    num_cols = 4
    cols = container.columns(num_cols)
    
    for idx, label in enumerate(ALL_LABELS):
        checkbox_key = f"cb_{i}_{label}"
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = (label in current_topics)

        cb_function = partial(on_checkbox_change, i, label)
        
        cols[idx % num_cols].checkbox(
            label,
            value=st.session_state[checkbox_key],
            key=checkbox_key,
            on_change=cb_function
        )

    container.markdown("</div>", unsafe_allow_html=True)

st.write("Changes are saved automatically to `updated/multilabel_test.csv` each time you toggle a checkbox.")
