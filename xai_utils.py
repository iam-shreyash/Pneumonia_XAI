import numpy as np
import tensorflow as tf
from PIL import Image
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2
from sklearn.metrics import jaccard_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_model(model_path):
    """Load the pre-trained CNN model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

def predict_image(model, sample_img):
    prob = model.predict(sample_img, verbose=0)[0][0]
    threshold = 0.65  
    label = "Pneumonia" if prob > threshold else "Normal"
    return label, prob

def compute_shap(model, sample_img, train_dir):
    """Compute SHAP values."""
    # Background data
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=100, class_mode='binary', shuffle=True)
    background = next(train_generator)[0][:100]
    
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(sample_img)
    return shap_values

def compute_lime(model, sample_img):
    """Compute LIME explanation."""
    sample_img_lime = (sample_img[0] * 255).astype(np.uint8)
    
    def predict_fn(images):
        processed = images.astype(np.float32) / 255.0
        probs = model.predict(processed, verbose=0)
        return np.hstack([1 - probs, probs])
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(sample_img_lime, predict_fn, top_labels=1, num_features=10000, num_samples=1000)
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=5, hide_rest=False)
    return explanation, mask, top_label

def overlay_heatmap(img, heatmap, alpha=0.5):
    """Overlay heatmap on image using OpenCV."""
    if heatmap.max() > 1:
        heatmap = heatmap / np.max(np.abs(heatmap))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed

def compute_iou(shap_heatmap, lime_heatmap):
    """Compute IoU between binary masks."""
    shap_mask = (shap_heatmap > np.percentile(shap_heatmap, 80)).astype(int).flatten()
    lime_mask = (lime_heatmap > 0).astype(int).flatten()
    if len(shap_mask) != len(lime_mask):
        return 0.0  # Fallback
    return jaccard_score(shap_mask, lime_mask, average='micro', zero_division=0)