import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from utils import evaluate_model

### Prepare dataset
# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

### Baseline model ###
# Loading/Training
keras_file = './models/mnist_baseline.h5'
if os.path.exists(keras_file):
  # Load existing model
  model = tf.keras.models.load_model(keras_file)
  model.summary()

  # recompile model for evaluation
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy'])
else:
  print('[E] Model not found.')

# # Evaluate baseline model
# _, baseline_model_accuracy = model.evaluate(
#     test_images, test_labels, verbose=1)

# print('Baseline test accuracy:', baseline_model_accuracy)

### Quantize model
quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

q_aware_model.summary()

train_images_subset = train_images[0:1000] # out of 60000
train_labels_subset = train_labels[0:1000]

q_aware_model.fit(train_images_subset, train_labels_subset,
                  batch_size=500, epochs=1, validation_split=0.1)

_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=1)

_, q_aware_model_accuracy = q_aware_model.evaluate(
   test_images, test_labels, verbose=1)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)

###
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

test_accuracy = evaluate_model(interpreter, test_images, test_labels)

print('Quant TFLite test_accuracy:', test_accuracy)
print('Quant TF test accuracy:', q_aware_model_accuracy)