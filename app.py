import streamlit as st
import torch
import numpy as np
import tempfile
import os
import cv2
from inference import (
    CricketModel,
    PoseExtractor,
    load_trained_model,
    load_reference_embeddings,
    classify_and_score
)
# ===== IMPORT YOUR BLOCK 3 CODE =====
# (Paste your full Block 3 code here OR import if modularized)

# Assume these are available from your code:
# - CricketModel
# - PoseExtractor
# - load_trained_model
# - load_reference_embeddings
# - classify_and_score

# ===== PATHS =====
MODEL_PATH = "best_model (3).pth"
EMBED_PATH = "reference_embeddings (2).npz"

# ===== LOAD ONCE (CACHED) =====
@st.cache_resource
def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_trained_model(MODEL_PATH)
    pose_extractor = PoseExtractor()
    ref_embeddings = load_reference_embeddings(EMBED_PATH)

    return model, pose_extractor, ref_embeddings

model, pose_extractor, ref_embeddings = load_all()

# ===== UI =====
st.title("🏏 Cricket Shot Classification")

st.write("Upload a cricket shot video to classify and score similarity.")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

metric = st.selectbox("Similarity Metric", ["cosine", "euclidean"])

if uploaded_file is not None:

    # Save temp video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("🚀 Run Inference"):

        with st.spinner("Processing video..."):

            result = classify_and_score(
                video_path,
                model,
                pose_extractor,
                ref_embeddings,
                metric
            )

        # ===== RESULTS =====
        st.success("✅ Done!")

        st.subheader("Prediction")
        st.write(f"**Class:** {result['predicted_class']}")
        st.write(f"**Confidence:** {result['confidence']:.4f}")

        st.subheader("Similarity Score")
        st.write(f"{result['similarity_score']:.4f} ({result['metric']})")

        # Clean up
        os.remove(video_path)