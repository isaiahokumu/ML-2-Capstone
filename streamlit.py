# streamlit_pneumonia_app.py
# Professional Pneumonia Detection App with Grad-CAM visualization
# Requirements: streamlit, tensorflow, pillow, numpy, matplotlib, seaborn

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image as kp_image

st.set_page_config(page_title="Pneumonia Detector", layout="centered")

# Utilities
@st.cache_resource(show_spinner=False)
def load_model(model_path_primary="best_pneumonia_model.keras", fallback_h5="best_pneumonia_model.h5"):
    """Load model with a fallback from .keras -> .h5. Cached to avoid reloading."""
    try:
        model = tf.keras.models.load_model(model_path_primary)
    except Exception:
        model = tf.keras.models.load_model(fallback_h5)
    return model


def find_last_conv_layer(model):
    """Find the deepest Conv2D layer even when nested (e.g., VGG inside a Functional model)."""
    last_conv = None

    def _recurse(layer):
        nonlocal last_conv
        # If layer is a model (nested) traverse its layers
        if hasattr(layer, 'layers') and layer.__class__.__name__ != 'Conv2D':
            for l in getattr(layer, 'layers'):
                _recurse(l)
        else:
            # Check by class name to avoid importing keras conv class
            if layer.__class__.__name__.lower().startswith('conv'):
                last_conv = layer

    for layer in model.layers:
        _recurse(layer)

    if last_conv is None:
        raise ValueError("No convolutional layer found in model")
    return last_conv.name


def get_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """Compute Grad-CAM heatmap for a single image array (1, H, W, C)."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Resolve nested names like 'vgg16/block5_conv3'
    target_layer = model
    if '/' in last_conv_layer_name:
        for part in last_conv_layer_name.split('/'):
            target_layer = target_layer.get_layer(part)
    else:
        # try direct, else search nested
        try:
            target_layer = model.get_layer(last_conv_layer_name)
        except ValueError:
            # search nested
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    try:
                        target_layer = layer.get_layer(last_conv_layer_name)
                        break
                    except Exception:
                        continue

    grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])

    # Ensure inputs are list-wrapped when calling model (avoids Keras warning)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array])
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(img_array_uint8, heatmap, alpha=0.4):
    """Overlay heatmap (H', W') onto original image (H, W, 3) and return float image in [0,1].
    img_array_uint8: HxWx3 uint8
    heatmap: 2D array scaled [0..1]
    """
    import matplotlib
    # map heatmap to colors
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    # Resize heatmap to image size
    jet_heatmap = tf.image.resize(tf.convert_to_tensor(jet_heatmap), (img_array_uint8.shape[0], img_array_uint8.shape[1]))
    jet_heatmap = np.array(jet_heatmap)

    # Combine
    img_float = img_array_uint8.astype('float32') / 255.0
    superimposed = np.clip(jet_heatmap * alpha + img_float, 0, 1)
    return superimposed


# App Layout
st.title("Pneumonia Detector — Chest X-Ray")
st.markdown("This demo classifies chest x-rays as **Normal** or **Pneumonia** and shows Grad-CAM visual explanations.")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    model_path = st.text_input("Model path (.keras or .h5)", value="best_pneumonia_model.keras")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)
    show_raw = st.checkbox("Show raw preprocessed image", value=False)
    show_heatmap_default = st.checkbox("Show heatmap by default", value=True)
    st.markdown("---")
    st.markdown("**Dataset quick stats (example)**")
    st.write("Train: ~5216 images\nTest: ~624 images")
    st.markdown("---")
    st.write("*For educational use only — not a clinical tool.*")

# Load model (cached)
with st.spinner("Loading model..."):
    model = load_model(model_path, fallback_h5="best_pneumonia_model.h5")

# Tabs
tab1, tab2, tab3 = st.tabs(["Home", "Predict", "Class"])

with tab1:
    st.subheader("How to use")
    st.markdown("1. Upload a chest x-ray image (jpg/png).\n2. Go to Predict tab to get probability and decision.\n3. In Grad-CAM tab you can visualize model attention.")

with tab2:
    st.subheader("Upload & Predict")
    uploaded = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"]) 

    if uploaded is not None:
        # Read image
        image_data = uploaded.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        original_size = img.size

        # Show original
        st.image(img, caption="Uploaded image (original)", use_container_width=True)

        # Preprocess for model
        input_size = (150, 150)
        img_resized = img.resize(input_size)
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array_batch = np.expand_dims(img_array, axis=0)

        if show_raw:
            st.image(img_resized, caption="Preprocessed (model input)", width=256)

        # Predict
        with st.spinner("Running model inference..."):
            preds = model.predict(img_array_batch)
            prob = float(preds[0][0]) if preds.shape[-1] == 1 else float(preds[0][1]) if preds.shape[-1] > 1 else float(preds[0])

        st.metric("Pneumonia probability", f"{prob*100:.2f}%")
        if prob >= threshold:
            st.error("Model suggests Pneumonia — consult a clinician")
        else:
            st.success("Model suggests Normal")

        # Option to compute Grad-CAM right away
        if st.button("Compute Grad-CAM") or show_heatmap_default:
            try:
                last_conv_name = None
                # prefer common nested naming
                if any('vgg16' in layer.name for layer in model.layers):
                    last_conv_name = 'vgg16/block5_conv3'

                heatmap = get_gradcam_heatmap(img_array_batch, model, last_conv_name)
                # heatmap is small (e.g., 4x4) -> resize handled in overlay function
                img_uint8 = np.array(img.resize((img.width, img.height))).astype('uint8')
                overlay = overlay_heatmap_on_image(img_uint8, heatmap)

                st.subheader("Grad-CAM explanation")
                st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)

            except Exception as e:
                st.error(f"Could not compute Grad-CAM: {e}")

with tab3:
    st.subheader("Batch predictions & EDA")
    st.markdown("Upload a zip of images or point to a local dataset directory for batch inference (advanced users).")

    # Small EDA example (static) - visualize class distribution
    st.write("Dataset balance example")
    fig, ax = plt.subplots(figsize=(5,3))
    sns.barplot(x=["NORMAL", "PNEUMONIA"], y=[1341, 3875], ax=ax)
    ax.set_ylabel("Number of images")
    st.pyplot(fig)

st.markdown("---")
st.caption("Model created for demonstration. Always include clinical validation before any medical use.")

# End of file
