import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight  # FIXED: For class weights
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths (your setup)
TRAIN_DIR = r'C:\Users\ASUS\OneDrive\Desktop\Broskies Internship AIML\pneumonia_xai_app\data\train'
TEST_DIR = r'C:\Users\ASUS\OneDrive\Desktop\Broskies Internship AIML\pneumonia_xai_app\data\test'
MODEL_PATH = 'pneumonia_model_balanced.h5'
THRESHOLD_PATH = 'optimal_threshold.txt'  

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,     
    width_shift_range=0.15, 
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)


train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224, 224), batch_size=32, class_mode='binary')
test_gen = test_datagen.flow_from_directory(TEST_DIR, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)

class_counts = np.bincount(train_gen.classes)  # [Normal_count, Pneumonia_count]
class_weights = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
class_weight_dict = dict(zip(np.unique(train_gen.classes), class_weights))
print(f"Class Distribution (Train): Normal={class_counts[0]}, Pneumonia={class_counts[1]}")
print(f"Class Weights: {class_weight_dict}")  # E.g., {0: 1.5, 1: 0.9} (Normal gets higher weight)

# ResNet50 Transfer Learning 
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1)
]


print("Training with class weights for imbalance...")
history = model.fit(
    train_gen,
    epochs=25,
    validation_data=test_gen,
    callbacks=callbacks,
    class_weight=class_weight_dict  
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Full predictions for report
y_pred_prob = model.predict(test_gen)
y_pred_classes = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=['Normal', 'Pneumonia']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# Tune Threshold with ROC 
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)
optimal_idx = np.argmax(tpr - fpr)  
optimal_threshold = thresholds[optimal_idx]
print(f"\nOptimal Threshold (for app): {optimal_threshold:.4f} (AUC: {roc_auc:.4f})")

# Save threshold for app
with open(THRESHOLD_PATH, 'w') as f:
    f.write(f"{optimal_threshold}")

# Plot ROC and History
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve'); plt.legend(loc="lower right")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.savefig('training_results.png', dpi=150)
plt.show()

# Save model
model.save(MODEL_PATH)
print(f"\nBalanced Model Saved: {MODEL_PATH}")
print(f"Use threshold {optimal_threshold:.4f} in app for best balance.")