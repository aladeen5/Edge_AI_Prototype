Edge AI Waste Classification using TensorFlow Lite
Overview

This project demonstrates a complete Edge AI pipeline for classifying recyclable waste items using a lightweight deep-learning model optimized with TensorFlow Lite.
It is designed to run efficiently on low-power devices such as Raspberry Pi, Jetson Nano, or any embedded system.

The system recognizes the following categories:

Cardboard

Glass

Metal

Paper

Plastic

Trash

 Project Goals
✔ Train a lightweight image classification model

Using MobileNetV2 as the backbone for efficient edge deployment.

✔ Convert model to TensorFlow Lite

Optimized for small size, low-memory, and fast inference.

✔ Run inference on sample images

Verify accuracy and real-time performance using TFLite Interpreter.

✔ Demonstrate Edge AI benefits

Explain why Edge AI is ideal for real-time applications like waste sorting and smart recycling bins.

 Dataset

The dataset used is TrashNet (Resized):

 Download:
https://zenodo.org/records/3439273/files/dataset-resized.zip?download=1

After downloading, the dataset is split automatically into:

train/ (70%)
val/   (20%)
test/  (10%)
 Installation
1️ Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

2️ Install Dependencies
pip install tensorflow tensorflow-hub pillow matplotlib numpy

 Model Training (train.py)

Training uses transfer learning with MobileNetV2.

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False


Model is trained for 10–15 epochs and saved as:

waste_classifier.h5

 TensorFlow Lite Conversion

After training:

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()


This produces:

waste_classifier.tflite

 Deployment on Raspberry Pi

Install TensorFlow Lite runtime:

pip install tflite-runtime


Copy your model:

scp waste_classifier.tflite pi@raspberrypi:/home/pi/


Run inference using the same inference.py script.

 Results

Model accuracy: 80–90% depending on dataset quality

TFLite model size: ~3–4 MB

Inference speed on Pi 4: 30–50 ms per image

 Author

Adoh Baraza (aladeen5)
