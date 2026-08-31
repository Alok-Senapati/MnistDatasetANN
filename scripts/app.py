"""Interactive Streamlit application for handwritten digit recognition with live canvas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from mnistdatasetann.explainability import (
    GradCAM,
    compute_saliency_map,
    visualize_explanations,
)
from mnistdatasetann.model import load_model
from mnistdatasetann.utils import preprocess_image, visualize_feature_maps

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
        help="Automatically inverts dark ink on light background to match MNIST format.",
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


def render_prediction_dashboard(
    raw_image: Image.Image,
    model: torch.nn.Module,
    auto_invert: bool,
    preprocess_method: str,
    device: str,
) -> None:
    """Run model inference on the input image and render the visual prediction dashboard."""
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
        st.image(raw_image, caption="Input Image", use_container_width=True)
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
            {"Probability (%)": probs * 100.0},
            index=[f"Digit {i}" for i in range(10)],
        )
        st.bar_chart(df_probs, y="Probability (%)")

    if hasattr(model, "get_feature_maps"):
        with st.expander("🔍 View Convolutional Feature Maps (Layer Activations)", expanded=False):
            st.markdown(
                "Visualizing intermediate channel activations captured by each convolutional block:"
            )
            fmaps = model.get_feature_maps(input_tensor[:1])
            fig_fmaps = visualize_feature_maps(
                feature_maps=fmaps,
                raw_image=feature_array[0],
                max_channels_per_layer=16,
            )
            st.pyplot(fig_fmaps)

    with st.expander("🧠 Explain Prediction (Saliency & Attention Heatmaps)", expanded=False):
        st.markdown(
            "Visualizing the exact pixel regions and convolutional filter attention "
            "that contributed to this prediction:"
        )
        saliency = compute_saliency_map(model, input_tensor[:1], target_class=predicted_digit)

        gradcam_map = None
        if hasattr(model, "conv_layers"):
            try:
                gradcam = GradCAM(model)
                gradcam_map = gradcam.generate_cam(input_tensor[:1], target_class=predicted_digit)
                gradcam.remove_hooks()
            except Exception:
                gradcam_map = None

        fig_explain = visualize_explanations(
            raw_image=feature_array[0],
            saliency_map=saliency,
            gradcam_map=gradcam_map,
            predicted_class=predicted_digit,
            confidence=confidence,
        )
        st.pyplot(fig_explain)


def main() -> None:
    """Run the Streamlit frontend for interactive digit classification."""
    st.title("🔢 MNIST Handwritten Digit Classifier")
    st.markdown(
        "Draw a digit live in the interactive canvas or upload an image to see "
        "real-time preprocessing, model inference, and probability distributions."
    )

    checkpoints = find_available_checkpoints()
    model, auto_invert, preprocess_method, device = render_sidebar(checkpoints)

    if model is None:
        return

    tab_canvas, tab_upload = st.tabs(["✍️ Live Drawing Canvas", "📁 Upload Image"])

    with tab_canvas:
        st.markdown("Draw a digit (0–9) below. The model will classify it in real time:")

        col_canvas, col_settings = st.columns([1.2, 1])

        with col_settings:
            stroke_width = st.slider("Stroke Width", min_value=8, max_value=32, value=18)
            dark_canvas = st.checkbox("Dark Canvas (White Ink on Black)", value=True)
            bg_color = "#000000" if dark_canvas else "#FFFFFF"
            stroke_color = "#FFFFFF" if dark_canvas else "#000000"
            st.caption("Use the trash bin icon on the bottom toolbar to clear the canvas.")

        with col_canvas:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="interactive_digit_canvas",
                display_toolbar=True,
            )

        if canvas_result.image_data is not None:
            img_arr = canvas_result.image_data.astype(np.uint8)
            # Check if any strokes have been drawn on canvas
            has_strokes = False
            if dark_canvas and np.any(img_arr[:, :, :3] > 40):
                has_strokes = True
            elif not dark_canvas and np.any(img_arr[:, :, :3] < 210):
                has_strokes = True

            if has_strokes:
                drawn_image = Image.fromarray(img_arr)
                render_prediction_dashboard(
                    drawn_image, model, auto_invert, preprocess_method, device
                )

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a digit image (drawn or photographed)",
            type=["png", "jpg", "jpeg", "webp"],
            key="file_uploader_widget",
        )

        if uploaded_file is not None:
            raw_image = Image.open(uploaded_file)
            render_prediction_dashboard(raw_image, model, auto_invert, preprocess_method, device)


if __name__ == "__main__":
    main()
