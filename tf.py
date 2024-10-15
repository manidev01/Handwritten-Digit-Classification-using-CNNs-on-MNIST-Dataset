import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Load and Preprocess the Data
# Loading the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 2. Data Preprocessing
# Normalize the image data (from range [0, 255] to [0, 1])
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the images to add a channel dimension (necessary for CNNs)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# 3. Exploratory Data Analysis (EDA)
# Visualize a few images and their labels
def plot_images(images, labels, n=5):
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()

plot_images(train_images, train_labels)

# 4. Build the CNN Model
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout to prevent overfitting
        layers.Dense(10, activation='softmax')  # 10 classes (digits 0-9)
    ])
    return model

cnn_model = create_cnn_model()

# 5. Compile the Model
cnn_model.compile(
    optimizer='adam',  # Adam optimizer
    loss='sparse_categorical_crossentropy',  # Suitable for integer-encoded labels
    metrics=['accuracy']
)

# 6. Split the Data
# For MNIST, we already have train/test sets, but for custom datasets, split them using sklearn's train_test_split

# 7. Train the Model
history = cnn_model.fit(train_images, train_labels, epochs=10, 
                        validation_split=0.2,  # Use 20% of training data for validation
                        batch_size=64)

# 8. Evaluate the Model
test_loss, test_acc = cnn_model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')

# 9. Model Evaluation Metrics
# Predict class probabilities
y_pred = cnn_model.predict(test_images)

# Convert predictions from probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(test_labels, y_pred_classes)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification Report (Precision, Recall, F1-score)
report = classification_report(test_labels, y_pred_classes)
print(report)

# 10. Improve the Model (Optional)
# You can modify the architecture, add more layers, increase filters, apply data augmentation, etc.

# 11. Visualize Predictions
# Visualize a few test images and their predicted labels
def plot_predictions(images, true_labels, pred_labels, n=5):
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'True: {true_labels[i]}, Pred: {pred_labels[i]}')
        plt.axis('off')
    plt.show()

plot_predictions(test_images, test_labels, y_pred_classes)

# 12. Save the Model
cnn_model.save('cnn_mnist_model.h5')
print("Model saved!")

# 13. Conclusion
# Evaluate how well the model performed, analyze misclassified samples and consider improvements.
