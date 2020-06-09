import tempfile
import os

import tensorflow as tf
import numpy as np

from tensorflow import keras
import tensorflow_model_optimization as tfmot
from utils import get_gzipped_model_size, evaluate_model

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
  # Define the model architecture.
  model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Reshape(target_shape=(28, 28, 1)),
      keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(10)
  ])

  # Train the digit classification model
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy'])

  model.fit(
      train_images,
      train_labels,
      epochs=6,
      validation_split=0.1,
  )

  # Save baseline model
  tf.keras.models.save_model(model, keras_file, include_optimizer=False)
  print('Saved baseline model to:', keras_file)

# Evaluate baseline model
_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)


### Pruning ###
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 4
validation_split = 0.1  # 10% of training set will be used for validation set.

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                             final_sparsity=0.70,
                                                             begin_step=0,
                                                             end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(
                              from_logits=True),
                          metrics=['accuracy'])

model_for_pruning.summary()

logdir = './logs/'

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(train_images, train_labels,
                      batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                      callbacks=callbacks)

# Compare accuracy
_, model_for_pruning_accuracy = model_for_pruning.evaluate(
    test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Pruned test accuracy:', model_for_pruning_accuracy)


model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)


### Create 3x smaller models for pruning ###
pruned_keras_file = './models/mnist_pruned.h5'

tf.keras.models.save_model(
    model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

pruned_tflite_file = './models/mnist_pruned.tflite'

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)

print("Size of gzipped baseline Keras model: %.2f bytes" %
      (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned Keras model: %.2f bytes" %
      (get_gzipped_model_size(pruned_keras_file)))
print("Size of gzipped pruned TFlite model: %.2f bytes" %
      (get_gzipped_model_size(pruned_tflite_file)))


### Create 10x smaller models by combining pruning and quantization ###
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

quantized_and_pruned_tflite_file = './models/mnist_qnp.tflite'

with open(quantized_and_pruned_tflite_file, 'wb') as f:
  f.write(quantized_and_pruned_tflite_model)

print('Saved quantized and pruned TFLite model to:',
      quantized_and_pruned_tflite_file)
print("Size of gzipped baseline Keras model: %.2f bytes" %
      (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" %
      (get_gzipped_model_size(quantized_and_pruned_tflite_file)))

### Evaluate TFLite models
interpreter = tf.lite.Interpreter(
    model_content=quantized_and_pruned_tflite_model)
interpreter.allocate_tensors()

test_accuracy = evaluate_model(interpreter, test_images, test_labels)

print('Pruned and quantized TFLite test_accuracy:', test_accuracy)
print('Pruned TF test accuracy:', model_for_pruning_accuracy)
