import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# Import utils
from xai_utils import load_model, predict_image, compute_shap, compute_lime, overlay_heatmap, compute_iou

# Config: Update paths if your setup differs
MODEL_PATH = 'pneumonia_model_balanced.h5'  # Ensure this is the balanced model
TRAIN_DIR = r'C:\Users\ASUS\OneDrive\Desktop\Broskies Internship AIML\pneumonia_xai_app\data\train'
st.set_page_config(page_title="Pneumonia XAI Demo", page_icon="🩺", layout="wide")

# Title and Disclaimer
st.title("🩺 Explainable AI for Pneumonia Detection in Chest X-Rays")
st.markdown("""
**Project Demo**: Upload a chest X-ray to get a CNN prediction (Normal/Pneumonia) with SHAP and LIME explanations.  
Heatmaps highlight influential regions (e.g., lung opacities). Compare methods for consistency (IoU metric).  
*Disclaimer: Educational prototype only—not for clinical use. Consult medical professionals for diagnosis.*
""")

# Sidebar for options
st.sidebar.header("Options")
compute_xai = st.sidebar.checkbox("Compute XAI Explanations (SHAP + LIME)", value=True)
alpha = st.sidebar.slider("Overlay Transparency (Alpha)", 0.1, 1.0, 0.4)
shap_method = st.sidebar.selectbox("Focus on SHAP Method", ["Both", "SHAP Only", "LIME Only"])

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-Ray (JPEG/PNG)", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)  # FIXED: use_container_width
    
    # Resize to model input (224x224)
    image_resized = image.resize((224, 224))
    sample_img = np.expand_dims(np.array(image_resized) / 255.0, axis=0)  # (1, 224, 224, 3)
    
    # Load model (cached)
    if 'model' not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.model = load_model(MODEL_PATH)
    model = st.session_state.model
    
    # Predict
    with st.spinner("Predicting..."):
        label, prob = predict_image(model, sample_img)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", label)
    with col2:
        st.metric("Confidence", f"{prob:.2%}")
    
    # FIXED: Low confidence warning
    if 0.4 < prob < 0.6:
        st.warning("Low confidence prediction—image may be unclear or borderline. Try a higher-quality X-ray.")
    
    # Original image for overlays
    original_img_uint8 = (sample_img[0] * 255).astype(np.uint8)
    
    if compute_xai:
        # Compute XAI (cached per session)
        if 'shap_values' not in st.session_state or 'explanation' not in st.session_state:
            with st.spinner("Computing SHAP and LIME (may take 2-5 mins)..."):
                try:
                    st.session_state.shap_values = compute_shap(model, sample_img, TRAIN_DIR)
                    st.session_state.explanation, st.session_state.mask, st.session_state.top_label = compute_lime(model, sample_img)
                except Exception as e:
                    st.error(f"XAI computation failed: {e}. Check data/train path and model. Skipping visuals.")
                    st.session_state.shap_values = None
                    st.session_state.explanation = None
        
        shap_values = st.session_state.shap_values
        explanation = st.session_state.explanation
        
        if shap_values is not None and explanation is not None:
            mask = st.session_state.mask
            top_label = st.session_state.top_label
            
            # Prepare heatmaps and overlays
            shap_heatmap = np.abs(shap_values[0][0]).max(axis=2)
            shap_overlay = overlay_heatmap(original_img_uint8, shap_heatmap, alpha)
            
            # LIME heatmap (robust)
            if len(mask.shape) == 3:
                lime_heatmap = np.mean(mask, axis=-1) > 0
            else:
                lime_heatmap = mask > 0
            lime_heatmap = lime_heatmap.astype(np.float32)
            # Weighted if possible
            local_exp = explanation.local_exp[top_label]
            if local_exp:
                try:
                    segments = explanation.segments
                    weighted_heatmap = np.zeros_like(segments, dtype=np.float32)
                    for seg_id, weight in local_exp:
                        if weight > 0:
                            weighted_heatmap[segments == seg_id] = weight
                    lime_heatmap = weighted_heatmap
                except:
                    pass
            lime_overlay = overlay_heatmap(original_img_uint8, lime_heatmap, alpha)
            
            # IoU
            iou = compute_iou(shap_heatmap, lime_heatmap)
            
            # Display results
            st.subheader("XAI Explanations")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("SHAP Heatmap")
                fig_shap, ax_shap = plt.subplots(figsize=(5, 5))
                im_shap = ax_shap.imshow(shap_heatmap, cmap='jet')
                ax_shap.set_title("SHAP (Pixel Influence)")
                ax_shap.axis('off')
                plt.colorbar(im_shap, ax=ax_shap)
                st.pyplot(fig_shap)
                plt.close(fig_shap)  # FIXED: Close to prevent blank outputs
                
                st.subheader("SHAP Overlay")
                st.image(shap_overlay, caption="SHAP Overlay (Red: High Influence)", use_container_width=True)  # FIXED: use_container_width
            
            with col2:
                st.subheader("LIME Heatmap")
                fig_lime, ax_lime = plt.subplots(figsize=(5, 5))
                im_lime = ax_lime.imshow(lime_heatmap, cmap='jet')
                ax_lime.set_title("LIME (Superpixel Influence)")
                ax_lime.axis('off')
                plt.colorbar(im_lime, ax=ax_lime)
                st.pyplot(fig_lime)
                plt.close(fig_lime)  # FIXED: Close to prevent blank outputs
                
                st.subheader("LIME Overlay")
                st.image(lime_overlay, caption="LIME Overlay (Red: Positive Contribution)", use_container_width=True)  # FIXED: use_container_width
            
            with col3:
                st.subheader("Comparison")
                fig_cmp, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.imshow(original_img_uint8); ax1.set_title("Original"); ax1.axis('off')
                ax2.imshow(shap_overlay); ax2.set_title(f"Overlay Comparison\n(IoU: {iou:.4f})"); ax2.axis('off')
                st.pyplot(fig_cmp)
                plt.close(fig_cmp)  # FIXED: Close to prevent blank outputs
                
                st.metric("Consistency (IoU)", f"{iou:.4f}")
                st.markdown(f"**Interpretation**: IoU >0.5 indicates SHAP/LIME agree on key regions (e.g., lungs). Red highlights show model's 'attention' for {label} prediction.")
            
            # Method selection (if "Only" chosen)
            if shap_method != "Both":
                st.info(f"Focused on {shap_method} as selected.")
        else:
            st.warning("XAI visuals unavailable—check console for errors.")
    
    else:
        st.info("XAI skipped. Upload and enable checkbox for explanations.")
    
    # Download results (optional - saves a sample plot)
    if st.button("Download Sample Report Plot"):
        fig_report, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(original_img_uint8); ax.set_title(f"XAI Summary: {label} (Confidence: {prob:.2%})")
        ax.axis('off')
        plt.savefig('xai_report.png', dpi=150, bbox_inches='tight')
        plt.close(fig_report)  # FIXED: Close
        with open('xai_report.png', 'rb') as f:
            st.download_button("Download PNG", f.read(), file_name="xai_report.png")

else:
    st.info("Upload an image to start.")
    st.markdown("**Demo Tip**: Use a chest X-ray (e.g., from data/test/PNEUMONIA) for best results.")