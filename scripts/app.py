"""Interactive Streamlit application for handwritten digit recognition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from mnistdatasetann.model import load_model
from mnistdatasetann.utils import preprocess_image

BASE_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)


def find_available_checkpoints() -> dict[str, Path]:
    """Scan the artifacts directory and map descriptive names to checkpoint paths."""
    checkpoints: dict[str, Path] = {}
    if not BASE_ARTIFACT_DIR.exists():
        return checkpoints

    for run_dir in sorted(BASE_ARTIFACT_DIR.iterdir(), reverse=True):
        checkpoint_path = run_dir / "best_model.pt"
        if checkpoint_path.is_file():
            label = f"Run: {run_dir.name}"
            report_path = run_dir / "classification_report.json"
            if report_path.is_file():
                try:
                    rep = json.loads(report_path.read_text(encoding="utf-8"))
                    acc = rep.get("accuracy", None)
                    if acc is not None:
                        label += f" (Test Acc: {acc * 100:.2f}%)"
                except Exception:
                    pass
            checkpoints[label] = checkpoint_path

    return checkpoints


@st.cache_resource
def get_cached_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load and cache the trained MLP classifier in evaluation mode."""
    return load_model(checkpoint_path, device=device, eval_mode=True)


def render_sidebar(
    checkpoints: dict[str, Path],
) -> tuple[torch.nn.Module | None, bool, str, str]:
    """Render sidebar controls and return selected inference settings."""
    st.sidebar.header("⚙️ Configuration")

    if not checkpoints:
        st.sidebar.error("No trained model checkpoints found under artifacts/")
        st.sidebar.info("Train a model first using: `uv run python scripts/train.py`")

        return None, True, "otsu", "cpu"

    selected_label = st.sidebar.selectbox("Select Model Run", options=list(checkpoints.keys()))
    selected_path = checkpoints[selected_label]
    device = st.sidebar.selectbox(
        "Inference Device",
        options=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
        index=0,
    )

    auto_invert = st.sidebar.checkbox(
        "Auto-Invert Colors",
        value=True,
        help="Automatically inverts dark ink on light background"
        " to match MNIST (bright digits on dark backgrounds).",
    )

    preprocess_option = st.sidebar.radio(
        "Preprocessing Method",
        options=["Otsu (Real Photos / Shadows)", "Standard (Clean Digital / Paint)"],
        index=0,
        help="Otsu is best for camera photos and noisy paper backgrounds. "
        "Standard is best for clean digital drawings.",
    )
    preprocess_method = "otsu" if "Otsu" in preprocess_option else "standard"

    model = get_cached_model(str(selected_path), device=device)

    with st.sidebar.expander("🔍 Architecture Info", expanded=False):
        st.code(str(model), language="text")

    return model, auto_invert, preprocess_method, device


def main() -> None:
    """Run the Streamlit frontend for interactive digit classification."""
    st.title("🔢 MNIST Handwritten Digit Classifier")
    st.markdown(
        "Upload an image of a handwritten digit (any resolution or format) to see "
        "real-time preprocessing, model inference, and probability distributions."
    )

    checkpoints = find_available_checkpoints()
    model, auto_invert, preprocess_method, device = render_sidebar(checkpoints)

    if model is None:
        return

    uploaded_file = st.file_uploader(
        "Upload a digit image (drawn or photographed)", type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)

        feature_array, canvas_28x28 = preprocess_image(
            raw_image, auto_invert=auto_invert, method=preprocess_method
        )
        input_tensor = torch.tensor(feature_array, dtype=torch.float32).to(device)

        with torch.no_grad():
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_tensor)[0].cpu().numpy()
            else:
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_digit = int(np.argmax(probs))
        confidence = float(probs[predicted_digit]) * 100.0

        st.divider()

        col1, col2, col3 = st.columns([1, 1, 1.5])

        with col1:
            st.subheader("🖼️ Preprocessing")
            st.image(raw_image, caption="Uploaded Original", use_container_width=True)
            scaled_preview = canvas_28x28.resize((140, 140), Image.Resampling.NEAREST)
            st.image(scaled_preview, caption="28x28 Model Input", use_container_width=False)

        with col2:
            st.subheader("🎯 Prediction")
            st.metric(
                label="Predicted Digit",
                value=f"{predicted_digit}",
                delta=f"{confidence:.2f}% Confidence",
            )

            top3_idx = np.argsort(probs)[::-1][:3]
            st.markdown("**Top Candidates:**")
            for rank, idx in enumerate(top3_idx, 1):
                st.write(f"**#{rank} Digit {idx}**: `{(probs[idx] * 100):.2f}%`")

        with col3:
            st.subheader("📊 Class Probabilities")
            df_probs = pd.DataFrame(
                {"Probability (%)": probs * 100.0}, index=[f"Digit {i}" for i in range(10)]
            )
            st.bar_chart(df_probs, y="Probability (%)")


if __name__ == "__main__":
    main()
