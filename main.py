# Set TensorFlow log level to suppress most warnings and info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

# Import TensorFlow library
import tensorflow as tf
import keras
import time
from datetime import datetime

# Start timing the entire execution
start_time = time.time()


def save_model(model):
    # Save the TFLite model to disk
    with open('model.tflite', 'wb') as f:
        f.write(model)  # type: ignore

def custom_model():
    # Data augmentation layer
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ])

    # Build a simple Convolutional Neural Network (CNN) model
    return keras.models.Sequential([
        data_augmentation,  # Add data augmentation as the first layer
        keras.layers.Rescaling(1./255, input_shape=(64, 64, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # First max pooling layer: reduces spatial size by taking max over 2x2 regions
        # Output: 31x31x32
        keras.layers.MaxPooling2D(2, 2),
        # Second convolutional layer: 64 filters, 3x3 kernel, ReLU activation
        # Output: 29x29x64
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # Second max pooling layer
        # Output: 14x14x64
        keras.layers.MaxPooling2D(2, 2),


        # keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # # Second max pooling layer
        # # Output: 14x14x64
        # keras.layers.MaxPooling2D(2, 2),

        # Flatten the 3D output to 1D for the dense layers
        # This output a vector of 14*14*64 = 12544 elements
        keras.layers.Flatten(),
        # Dense (fully connected) layer with 64 units and ReLU activation
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),  # Add dropout for regularization
        keras.layers.Dense(1, activation='sigmoid')
    ])

def pretrained_model():

    base_model = keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),  # Use at least 96x96 for MobileNetV2
        include_top=False,
        weights='imagenet'
    )

    return keras.Sequential([
        keras.layers.Resizing(128, 128),  # Resize if your images are smaller
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')  # For binary classification
    ])


# Load images from the 'data/train' directory using modern approach
# - Expects subfolders for each class (e.g., 'dogs', 'cats')
# - image_size resizes all images to 64x64 pixels
# - batch_size determines how many images are loaded at a time
# - label_mode='binary' for two classes (dog/cat)

print(f"Starting data loading at {datetime.now().strftime('%H:%M:%S')}")
data_load_start = time.time()

train_dataset = keras.utils.image_dataset_from_directory(
    './data/cats_and_dogs_filtered/train',  # Path to training data directory
    image_size=(64, 64),     # Resize images to 64x64 pixels
    batch_size=32,           # Number of images per batch
    label_mode='binary'      # Binary labels (0 or 1)
)

test_dataset = keras.utils.image_dataset_from_directory(
    './data/cats_and_dogs_filtered/validation',  # Path to testing data directory
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary'
)

data_load_time = time.time() - data_load_start
print(f"Data loading completed in {data_load_time:.2f} seconds")

# model = custom_model()
model = pretrained_model()

# Compile the model
# - optimizer='adam' is a popular optimizer for training
# - loss='binary_crossentropy' is used for binary classification
# - metrics=['accuracy'] tracks accuracy during training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the training data generator
# - epochs=3 means the model will see the entire dataset 3 times
print(f"Starting model training at {datetime.now().strftime('%H:%M:%S')}")
training_start = time.time()

model.fit(train_dataset, epochs=5, validation_data=test_dataset)

training_time = time.time() - training_start
print(f"Model training completed in {training_time:.2f} seconds")
print(f"Total execution time: {data_load_time + training_time:.2f} seconds")

# Evaluate the model on the test dataset
# print('Testing model...')
# loss, accuracy = model.evaluate(test_dataset)
# print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')

# Convert the trained model to TensorFlow Lite format
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()  # type: ignore

# Calculate and print total execution time
total_time = time.time() - start_time
print(f'took {total_time:.2f} seconds to execute')

print(f'model summary: {model.summary()}')