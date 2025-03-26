import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from datasets import load_dataset
# Define the function to normalize the image and mask
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

# Define a function to load and preprocess the data
def load_image(datapoint):
    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


# Load the dataset
try:
    dataset = load_dataset("keremberke/satellite-building-segmentation", "full")
except tfds.core.registered.DatasetNotFoundError as e:
    print(f"Dataset not found.  Please verify the dataset name and ensure it's registered with TFDS. Error: {e}")
    exit()
except ValueError as e:
    print (f"ValueError encountered: {e}. Check that the 'full' split exists for the dataset, or you have necessary read permission")
    exit()
except Exception as e:
    print(f"An unexpected error: {e}")
    exit()

# Preprocess the dataset
BATCH_SIZE = 64
BUFFER_SIZE = 1000
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Define the DeepLabV3+ model
def deeplabv3_plus(input_shape, num_classes):
    model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [model.get_layer(name).output for name in layer_names]

    # Create the feature extractor model
    down_stack = tf.keras.Model(inputs=model.input, outputs=layers)
    down_stack.trainable = True

    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),   # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        num_classes, 3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def upsample(filters, size):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    return result


# Get the number of classes from the dataset info (if available), otherwise, assume 3.
num_classes = 3

# Create the model
model = deeplabv3_plus(input_shape=[256, 256, 3], num_classes=num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Train the model
EPOCHS = 10  # You can adjust the number of epochs
model_history = model.fit(dataset, epochs=EPOCHS)

#  Display the training progress (optional).
loss = model_history.history['loss']
accuracy = model_history.history['accuracy']

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), accuracy, label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# Make predictions on a few samples from the dataset (optional)
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset, num=1):
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])


def display(display_list):
  plt.figure(figsize=(15, 15))
  title = ['Input Image', 'True Mask', 'Predicted Mask']
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

show_predictions(dataset, 3)


