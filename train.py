import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# -------------------- Configuration --------------------
DATA_PATH = 'lung_cancer_clean_50k.csv' 
MODEL_DIR = 'models'
PLOTS_DIR = 'plots'
NOISE_STD_DEV = 0.90  # Standard deviation for Gaussian noise
L2_REG = 0.01         # L2 regularization factor
DROPOUT_RATE = 0.5    # Dropout rate
EPOCHS = 100          # Number of training epochs
BATCH_SIZE = 64       # Batch size for training
LEARNING_RATE = 0.001 # Learning rate for Adam optimizer
TOP_K_FEATURES = 15   # Number of top features to select
VAL_SPLIT = 0.2       # Validation split ratio from training data
SEED = 42             # Random seed for reproducibility

# Disable ONEDNN for consistent performance if not specifically needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow informational messages (1=errors, 2=errors+warnings, 3=errors+warnings+info)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------- Load and Preprocess Data --------------------
def load_and_prepare_data():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: '{DATA_PATH}' not found. Make sure the file is in the working directory or provide the correct path.")
    
    # Drop unnecessary columns if they exist
    df.drop(columns=['index', 'Patient Id'], inplace=True, errors='ignore')
    
    if 'Level' not in df.columns:
        raise ValueError("Error: Target column 'Level' not found in the dataset.")
    
    features_df = df.drop(columns=['Level'])
    labels_series = df['Level'].astype(int) # Ensure labels are integers

    # Select top-K features based on absolute correlation with labels
    # Ensure all feature columns are numeric for correlation calculation
    numeric_features_df = features_df.select_dtypes(include=np.number)
    if numeric_features_df.shape[1] < features_df.shape[1]:
        print("Warning: Non-numeric columns found and excluded from correlation-based feature selection.")

    corr_with_target = numeric_features_df.corrwith(labels_series).abs()
    selected_features = corr_with_target.sort_values(ascending=False).head(TOP_K_FEATURES).index.tolist()
    
    if not selected_features:
        raise ValueError("Error: No features were selected. Check TOP_K_FEATURES or data content.")
    
    print(f"✅ Selected Top {TOP_K_FEATURES} Features: {selected_features}")
    
    X = df[selected_features].values # Use .values to get NumPy array
    y = labels_series.values         # Use .values to get NumPy array
    
    return X, y, selected_features

# -------------------- Model Definition --------------------
def build_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,), name='Input_Layer'),
        Dense(16, activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(8, activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', # For one-hot encoded labels
                  metrics=['accuracy'])
    return model

# -------------------- Plotting Helpers --------------------
def plot_confusion_matrix(cm, class_names, filename="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()

def plot_training_history(history, filename_acc="training_accuracy.png", filename_loss="training_loss.png"):
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename_acc))
    plt.close()

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename_loss))
    plt.close()

def plot_multiclass_roc(y_true_cat, y_pred_prob, class_names, filename="roc_curve_multiclass.png"):
    # y_true_cat should be one-hot encoded
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_cat[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve for {class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)') # Dashed diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()

def plot_tsne(X_scaled, y_true_single_label, num_classes, filename="tsne_visualization.png"):
    # y_true_single_label should be integer labels (0, 1, 2, ...)
    try:
        perplexity_value = min(30.0, float(X_scaled.shape[0] - 1))
        if perplexity_value < 1.0: # t-SNE perplexity must be at least 1
            print("⚠️ Warning: Not enough samples for t-SNE plot (perplexity < 1). Skipping t-SNE.")
            return

        tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=SEED, learning_rate='auto', init='pca')
        X_embedded = tsne.fit_transform(X_scaled)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true_single_label, cmap='viridis', s=10, alpha=0.7)
        plt.colorbar(scatter, ticks=np.arange(num_classes), label='Class')
        plt.title("t-SNE Visualization of Test Data Embeddings")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename))
        plt.close()
    except Exception as e:
        print(f"❌ Error during t-SNE plotting: {e}")

# -------------------- Main Execution --------------------
def main():
    X, y, top_features = load_and_prepare_data()

    # Split dataset into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=SEED
    )

    print(f"Distribution before SMOTE (Train): {np.bincount(y_train)}")
    # Apply SMOTE on training data only to balance classes
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Distribution after SMOTE (Train): {np.bincount(y_train_res)}")

    # Add Gaussian noise to resampled training data and original test data
    # Noise is added *before* scaling
    X_train_noisy = X_train_res + np.random.normal(0, NOISE_STD_DEV, X_train_res.shape)
    X_test_noisy = X_test + np.random.normal(0, NOISE_STD_DEV, X_test.shape)

    # Normalize data using StandardScaler
    # Fit scaler on noisy training data, then transform both noisy train and noisy test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_noisy)
    X_test_scaled = scaler.transform(X_test_noisy) # Use the same scaler fitted on training data

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False, categories='auto') # sparse_output for dense array
    y_train_cat = encoder.fit_transform(y_train_res.reshape(-1, 1))
    y_test_cat = encoder.transform(y_test.reshape(-1, 1)) # Use the same encoder

    num_classes = y_train_cat.shape[1]
    class_names = [f'Class {i}' for i in range(num_classes)]

    # Build and compile model
    model = build_model(X_train_scaled.shape[1], num_classes)
    model.summary()

    # Early stopping callback to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)

    # Train the model
    print("\n🚀 Starting model training...")
    history = model.fit(
        X_train_scaled, y_train_cat,
        validation_split=VAL_SPLIT, # Uses a portion of X_train_scaled for validation
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2 # Shows one line per epoch
    )

    # Evaluate the model on the (noisy and scaled) test set from the initial split
    print("\n🧪 Evaluating model on the test set...")
    loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    print(f"\n✅ Test Accuracy: {accuracy*100:.2f}% | Test Loss: {loss:.4f}")

    # Predict probabilities and classes on the test set
    y_pred_probs = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred_probs, axis=1) # Get class indices
    y_true_classes = np.argmax(y_test_cat, axis=1)   # Get true class indices from one-hot

    # Print classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    # Generate and save plots
    print("\n🖼️ Generating plots...")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plot_confusion_matrix(cm, class_names)
    plot_training_history(history)
    plot_multiclass_roc(y_test_cat, y_pred_probs, class_names) # Pass one-hot true labels
    plot_tsne(X_test_scaled, y_true_classes, num_classes) # Pass single-label true labels

    # Save model, scaler, encoder, and the list of top features used for training
    print("\n💾 Saving model and preprocessing objects...")
    model.save(os.path.join(MODEL_DIR, "lung_cancer_model_final.keras"))
    with open(os.path.join(MODEL_DIR, "scaler_final.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "encoder_final.pkl"), 'wb') as f:
        pickle.dump(encoder, f)
    with open(os.path.join(MODEL_DIR, "training_features_final.pkl"), 'wb') as f:
        pickle.dump(top_features, f)
    
    print("\n🎯 Training complete.")
    print(f"✔ Features Used: {top_features}")
    print(f"✔ Noise Std Dev: {NOISE_STD_DEV}")
    print(f"✔ Model, scaler, encoder, and feature list saved to '{MODEL_DIR}' directory.")

if __name__ == "__main__":
    main()
