
# model_training.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Step 1: Check GPU/CPU availability ---
print("✅ TensorFlow version:", tf.__version__)
print("✅ Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# --- Step 2: Define dataset directories ---
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# --- Step 3: Data preprocessing and augmentation ---
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

# --- Step 4: Build the CNN model ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 5 emotion classes
])

# --- Step 5: Compile the model ---
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Step 6: Train the model ---
print("\n🚀 Training started... Please wait, this may take several minutes.\n")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=25
)

# --- Step 7: Create folder (optional) and save the model ---
os.makedirs('models', exist_ok=True)
model.save('models/face_emotionModel.h5')

print("\n✅ Model training completed and saved successfully as 'models/face_emotionModel.h5'")