import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_recall_fscore_support
from tensorflow.keras.models import load_model
import os

# --- PARAMETERS (Must match training) ---
NOISE_STD_DEV = 0.95   # Same noise std dev as used during training
# ----------------------------------------

# --- PATHS ---
MODEL_PATH = 'models/lung_cancer_model_final.keras'
SCALER_PATH = 'models/scaler_final.pkl'
FEATURES_PATH = 'models/training_features_final.pkl'
NEW_TEST_DATA_PATH = 'lung_cancer_test_50k.csv'  
PLOTS_DIR = 'plots_test_evaluation'
# --------------

# Create plots directory if needed
os.makedirs(PLOTS_DIR, exist_ok=True)

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"Loading model from: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print(f"Loading scaler from: {SCALER_PATH}")
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit()

print(f"Loading feature list from: {FEATURES_PATH}")
try:
    with open(FEATURES_PATH, 'rb') as f:
        top_features = pickle.load(f)
    print(f"Feature list loaded. Features: {top_features}")
except Exception as e:
    print(f"Error loading feature list: {e}")
    exit()

print(f"\nLoading new test data from: {NEW_TEST_DATA_PATH}")
try:
    new_df = pd.read_csv(NEW_TEST_DATA_PATH)
    print(f"New test data loaded. Shape: {new_df.shape}")
except FileNotFoundError:
    print(f"Test data file '{NEW_TEST_DATA_PATH}' not found.")
    exit()
except Exception as e:
    print(f"Error loading test data: {e}")
    exit()

# Drop ID columns if present
columns_to_drop_for_test = ['index', 'Patient Id']
new_df.drop(columns=[c for c in columns_to_drop_for_test if c in new_df.columns], inplace=True, errors='ignore')

if 'Level' not in new_df.columns:
    print(f"Target column 'Level' not found in test data. Available columns: {new_df.columns.tolist()}")
    exit()

# Check missing features
missing_features = [f for f in top_features if f not in new_df.columns]
if missing_features:
    print(f"Missing features in test data: {missing_features}")
    exit()

X_new_test_orig = new_df[top_features].values
y_new_test_true_labels = new_df['Level'].astype(int).values

# Print class distribution for info
print("Class distribution in test data:", np.bincount(y_new_test_true_labels))

# --- Preprocess test data ---
print(f"Adding Gaussian noise with std dev = {NOISE_STD_DEV}")
np.random.seed(42)
noise = np.random.normal(0, NOISE_STD_DEV, X_new_test_orig.shape)
X_new_test_noisy = X_new_test_orig + noise
print("Noise added.")

print("Scaling features using loaded scaler...")
X_new_test_scaled = scaler.transform(X_new_test_noisy)
print("Scaling done.")

# One-hot encode true labels to match model output
num_classes = model.output_shape[-1]
print(f"Model expects {num_classes} classes.")

y_new_test_cat = pd.get_dummies(y_new_test_true_labels).values

# Align one-hot shape if needed
if y_new_test_cat.shape[1] < num_classes:
    print("Aligning one-hot labels to match model output classes...")
    aligned = np.zeros((y_new_test_cat.shape[0], num_classes))
    present_classes = sorted(np.unique(y_new_test_true_labels))
    for i, cls in enumerate(present_classes):
        if cls < num_classes:
            aligned[:, cls] = y_new_test_cat[:, i]
    y_new_test_cat = aligned
    print(f"One-hot labels aligned. New shape: {y_new_test_cat.shape}")
elif y_new_test_cat.shape[1] > num_classes:
    print(f"Error: More classes in test labels ({y_new_test_cat.shape[1]}) than model output ({num_classes}).")
    exit()

# --- Predict ---
print("\nMaking predictions on test data...")
try:
    y_pred_probs = model.predict(X_new_test_scaled)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    print("Predictions done.")
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# --- Evaluate ---
print("\nEvaluating predictions:")

try:
    loss, acc = model.evaluate(X_new_test_scaled, y_new_test_cat, verbose=0)
    print(f"Keras evaluation -> Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")
except Exception as e:
    print(f"Error during model.evaluate: {e}")
    acc = None

acc_sklearn = accuracy_score(y_new_test_true_labels, y_pred_classes)
print(f"Sklearn accuracy: {acc_sklearn*100:.2f}%")

# Precision, Recall, F1 Score per class
precision, recall, f1, _ = precision_recall_fscore_support(y_new_test_true_labels, y_pred_classes, average=None)
for i in range(num_classes):
    print(f"Class {i}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1-score={f1[i]:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_new_test_true_labels, y_pred_classes)
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Test Data')
plt.colorbar()
classes = sorted(np.unique(y_new_test_true_labels))
plt.xticks(classes, [f"Class {i}" for i in classes], rotation=45)
plt.yticks(classes, [f"Class {i}" for i in classes])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix_test.png'))
plt.close()
print(f"Confusion matrix saved to {os.path.join(PLOTS_DIR, 'confusion_matrix_test.png')}")

# Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8,6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix - Test Data')
plt.colorbar()
plt.xticks(classes, [f"Class {i}" for i in classes], rotation=45)
plt.yticks(classes, [f"Class {i}" for i in classes])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix_normalized_test.png'))
plt.close()
print(f"Normalized confusion matrix saved to {os.path.join(PLOTS_DIR, 'confusion_matrix_normalized_test.png')}")

# ROC Curve for class 1 (example)
TARGET_CLASS_FOR_ROC = 1
if num_classes > 1 and TARGET_CLASS_FOR_ROC in classes:
    y_true_bin = (y_new_test_true_labels == TARGET_CLASS_FOR_ROC).astype(int)
    y_scores = y_pred_probs[:, TARGET_CLASS_FOR_ROC]
    fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve for Class {TARGET_CLASS_FOR_ROC} (area = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Class {TARGET_CLASS_FOR_ROC}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'roc_curve_class{TARGET_CLASS_FOR_ROC}_test.png'))
    plt.close()
    print(f"ROC curve saved to {os.path.join(PLOTS_DIR, f'roc_curve_class{TARGET_CLASS_FOR_ROC}_test.png')}")
else:
    print(f"ROC curve for Class {TARGET_CLASS_FOR_ROC} skipped (class not present or single class problem).")

# --- Sample predictions ---
print("\nSample Predictions (Actual vs Predicted):")
for i in range(min(10, len(y_new_test_true_labels))):
    print(f"Sample {i+1}: Actual = {y_new_test_true_labels[i]}, Predicted = {y_pred_classes[i]}")

print("\nTesting script finished.")
