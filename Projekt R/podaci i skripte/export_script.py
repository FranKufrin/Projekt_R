# %%
import cv2
import numpy as np
import datumaro as dm

import gc

gc.collect()


dataset = dm.Dataset.import_from('../fastener_dataset/annotations/instances_default.json', format='coco')
#stats = compute_ann_statistics(dataset)
#print(dataset)

dataset_iter = iter(dataset)
dataset_list = list(dataset)

# images = []
# labels = []
data = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
}
 
for i in range(0, 16970):
    print('processing image... ', i)
    item = dataset_list[i]
    item_annotation = item.annotations[0]
    
    new_path = item.media.path.replace(':', '_')
    # item_img = cv2.imread(new_path)
    # item_img = cv2.cvtColor(item_img, cv2.COLOR_BGR2RGB)
    # res = cv2.resize(item_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    
    item_label = item_annotation.as_dict()['label']
    
    # images.append(res)
    # labels.append(item_label)
    
    data[item_label].append(new_path)

# images = np.array(images)
# labels = np.array(labels)

print(data)
# print(labels.shape)


# %%
import random

# for i in data.keys():
#     random.shuffle(data[i])
    
import os, shutil, pathlib

new_base_dir = pathlib.Path("./data")
source_dir = pathlib.Path("../fastener_dataset/images")

for entry in data:
    dir = new_base_dir / str(entry)
    os.makedirs(dir)

    fnames = [f"{i.split('/')[-3]}/{i.split('/')[-2]}/{i.split('/')[-1]}" for i in data[entry]]
    print(fnames)
    
    for fname in fnames:
        shutil.copyfile(src = source_dir / pathlib.Path(fname),
                        dst = dir / fname.split('/')[-1])

# %%

import tensorflow as tf
import os
import random
import cv2
import numpy as np

def create_dataset_slice(count_train, count_validation, count_test):
    train_data = []
    validation_data = []
    test_data = []
    
    for i in range(0, 6):
        print('processing class... ', i)
        directory = f'./data/{str(i)}'
        files = os.listdir(directory)
        random.shuffle(files)
                
        train_files = files[:count_train]
        validation_files = files[count_train:count_train+count_validation]
        test_files = files[count_train+count_validation:count_train+count_validation+count_test]
        
        for file in train_files:
            new_path = f'./data/{str(i)}/{file.replace(":", "_")}'
            item_img = cv2.imread(new_path)
            item_img = cv2.cvtColor(item_img, cv2.COLOR_BGR2RGB)
            res = cv2.resize(item_img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            item_label = i
            train_data.append((res, item_label))
            
        for file in validation_files:
            new_path = f'./data/{str(i)}/{file.replace(":", "_")}'
            item_img = cv2.imread(new_path)
            item_img = cv2.cvtColor(item_img, cv2.COLOR_BGR2RGB)
            res = cv2.resize(item_img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            item_label = i
            validation_data.append((res, item_label))
            
        for file in test_files:
            new_path = f'./data/{str(i)}/{file.replace(":", "_")}'
            item_img = cv2.imread(new_path)
            item_img = cv2.cvtColor(item_img, cv2.COLOR_BGR2RGB)
            res = cv2.resize(item_img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            item_label = i
            test_data.append((res, item_label))
    
    random.shuffle(train_data)
    random.shuffle(validation_data)
    random.shuffle(test_data)

    return train_data, validation_data, test_data

train_set, validation_set, test_set = create_dataset_slice(400, 100, 100)

(train_images, train_labels) = zip(*train_set)
(validation_images, validation_labels) = zip(*validation_set)
(test_images, test_labels) = zip(*test_set)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

# %%
for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    
for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    
for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)

# %%
from tensorflow import keras
from keras import layers

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)

inputs = keras.Input(shape=(256, 256, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)

x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(6, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# %%
model.compile(loss="sparse_categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["accuracy"])

# %%
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch.keras",
        save_best_only=True,
        monitor="val_loss")
]

history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=validation_dataset,
        callbacks=callbacks)

# %%
import matplotlib.pyplot as plt
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# %%
test_model = keras.models.load_model("convnet_from_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


