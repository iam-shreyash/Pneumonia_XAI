import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths (update if needed)
MODEL_PATH = 'pneumonia_model_balanced.h5'
TEST_DIR = r'C:\Users\ASUS\OneDrive\Desktop\Broskies Internship AIML\pneumonia_xai_app\data\test'  # Your test folder

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(TEST_DIR, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)

# Evaluate
loss, accuracy = model.evaluate(test_gen, verbose=1)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Test Loss: {loss:.4f}")

# Per-class precision/recall (optional, needs sklearn)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Predictions
y_pred = model.predict(test_gen)
print(f"Probabilities Summary: Min={y_pred.min():.4f}, Max={y_pred.max():.4f}, Mean={y_pred.mean():.4f}")
print(f"Probs >0.5: {np.sum(y_pred > 0.5) / len(y_pred) * 100:.1f}%")
y_pred_classes = (y_pred > 0.5).astype(int).flatten()
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=['Normal', 'Pneumonia']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
