import os
import boto3
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# AWS S3 Configuration
s3 = boto3.client('s3')
bucket_name = 'your-s3-bucket-name'
data_folder = 'data/'

# Data Pipeline
def download_data_from_s3():
    s3.download_file(bucket_name, 'train_data.zip', 'train_data.zip')
    s3.download_file(bucket_name, 'test_data.zip', 'test_data.zip')

def extract_data():
    os.system('unzip train_data.zip')
    os.system('unzip test_data.zip')

def preprocess_data():
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        'train_data/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    test_data = test_datagen.flow_from_directory(
        'test_data/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    return train_data, test_data

# ML Pipeline
def train_model(train_data, test_data):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=10, validation_data=test_data)

    return model

# Deployment Pipeline
def save_model(model):
    model.save('model.h5')
    s3.upload_file('model.h5', bucket_name, 'model.h5')

# Inference Pipeline
def load_model():
    s3.download_file(bucket_name, 'model.h5', 'model.h5')
    return tf.keras.models.load_model('model.h5')

def predict_image(model, image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = image / 255.0

    prediction = model.predict(image)
    if prediction[0][0] >= 0.5:
        return 'Object Detected'
    else:
        return 'No Object Detected'

# Execute the pipeline
download_data_from_s3()
extract_data()
train_data, test_data = preprocess_data()
trained_model = train_model(train_data, test_data)
save_model(trained_model)
loaded_model = load_model()
prediction = predict_image(loaded_model, 'image.jpg')
print(prediction)
