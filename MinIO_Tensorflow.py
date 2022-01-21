# -*- coding: utf-8 -*-
# Copyright MinIO 2020
import math
import os
import random
import tarfile
import time
from datetime import datetime

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio  # pylint: disable=W0611
from dotenv import load_dotenv
from minio import Minio
from minio.error import InvalidResponseError
from tensorflow import keras

RANDOM_SEED = 44
BATCH_SIZE = 128
PREPROCESSED_DATA_FOLDER = "preprocessed-data"
TF_RECORD_FILE_SIZE = 500

load_dotenv()
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_USE_HTTPS = bool(int(os.getenv("S3_USE_HTTPS")))
DATASET_BUCKET = os.getenv("DATASET_BUCKET")

minioClient = Minio(
    S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=S3_USE_HTTPS,
)

try:
    minioClient.fget_object(DATASET_BUCKET, "aclImdb_v1.tar.gz", "/tmp/dataset.tar.gz")
except InvalidResponseError as err:  # pylint: disable=W0621
    print(err)

extract_folder = f"/tmp/{DATASET_BUCKET}/"

with tarfile.open("/tmp/dataset.tar.gz", "r:gz") as tar:
    tar.extractall(path=extract_folder)

# Pre-Processing

train = []
test = []

dirs_to_read = [
    "aclImdb/train/pos",
    "aclImdb/train/neg",
    "aclImdb/test/pos",
    "aclImdb/test/neg",
]

for dir_name in dirs_to_read:
    parts = dir_name.split("/")
    dataset = parts[1]
    label = parts[2]
    for filename in os.listdir(os.path.join(extract_folder, dir_name)):
        with open(os.path.join(extract_folder, dir_name, filename), "r") as f:
            content = f.read()
            if dataset == "train":
                train.append({"text": content, "label": label})
            elif dataset == "test":
                test.append({"text": content, "label": label})

random.Random(RANDOM_SEED).shuffle(train)
random.Random(RANDOM_SEED).shuffle(test)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


def _embedded_sentence_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _label_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def encode_label(label):
    ret_val = None
    if label == "pos":
        ret_val = tf.constant([1, 0])
    elif label == "neg":
        ret_val = tf.constant([0, 1])

    if ret_val is None:
        raise ValueError(
            f"'{label}' is not a valid label (only 'pos' or 'neg' are allowed)"
        )
    return ret_val


def serialize_example(label, sentence_tensor):
    feature = {
        "sentence": _embedded_sentence_feature(sentence_tensor[0]),
        "label": _label_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto


def process_examples(records, prefix=""):
    starttime = time.time()
    total_training = len(records)
    print(f"Total of {total_training} elements")
    total_batches = math.floor(total_training / TF_RECORD_FILE_SIZE)
    if total_training % TF_RECORD_FILE_SIZE != 0:
        total_batches += 1
    print(f"Total of {total_batches} files of {TF_RECORD_FILE_SIZE} records")

    counter = 0
    file_counter = 0
    buffer = []
    file_list = []
    for elem in records:
        counter += 1

        sentence_embedding = embed([elem["text"]])
        label_encoded = encode_label(elem["label"])
        record = serialize_example(label_encoded, sentence_embedding)
        buffer.append(record)

        if counter >= TF_RECORD_FILE_SIZE:
            print(f"Records in buffer {len(buffer)}")
            # save this buffer of examples as a file to MinIO
            counter = 0
            file_counter += 1
            file_name = f"{prefix}_file{file_counter}.tfrecord"
            with open(file_name, "w+") as file:
                with tf.io.TFRecordWriter(file.name) as writer:
                    for example in buffer:
                        writer.write(example.SerializeToString())
            print(f"file size {os.stat(file_name).st_size}")
            try:
                minioClient.fput_object(
                    DATASET_BUCKET, f"{PREPROCESSED_DATA_FOLDER}/{file_name}", file_name
                )
            except InvalidResponseError as err:  # pylint: disable=W0621
                print(err)
            file_list.append(file_name)
            os.remove(file_name)
            buffer = []
            print(
                f"Done with batch {file_counter}/{total_batches} - {time.time() - starttime}"
            )
    if len(buffer) > 0:
        file_counter += 1
        file_name = f"{prefix}_file{file_counter}.tfrecord"
        with open(file_name, "w+") as file:
            with tf.io.TFRecordWriter(file.name) as writer:
                for example in buffer:
                    writer.write(example.SerializeToString())
        try:
            minioClient.fput_object(
                DATASET_BUCKET, f"{PREPROCESSED_DATA_FOLDER}/{file_name}", file_name
            )
        except InvalidResponseError as err:  # pylint: disable=W0621
            print(err)
        file_list.append(file_name)
        os.remove(file_name)
        buffer = []
    print("total time is :", time.time() - starttime)
    return file_list


process_examples(train, prefix="train")
process_examples(test, prefix="test")
print("Done!")

# List all training tfrecord files
objects = minioClient.list_objects(
    DATASET_BUCKET, prefix=f"{PREPROCESSED_DATA_FOLDER}/train"
)
training_files_list = []
for obj in objects:
    training_files_list.append(obj.object_name)

# List all testing tfrecord files
objects = minioClient.list_objects(
    DATASET_BUCKET, prefix=f"{PREPROCESSED_DATA_FOLDER}/test"
)
testing_files_list = []
for obj in objects:
    testing_files_list.append(obj.object_name)

all_training_filenames = [f"s3://{DATASET_BUCKET}/{f}" for f in training_files_list]
testing_filenames = [f"s3://{DATASET_BUCKET}/{f}" for f in testing_files_list]

total_train_data_files = math.floor(len(all_training_filenames) * 0.9)
if total_train_data_files == len(all_training_filenames):
    total_train_data_files -= 1
training_files = all_training_filenames[0:total_train_data_files]
validation_files = all_training_filenames[total_train_data_files:]

# Due to an uncatched TypeError caused by urllib3(==1.26.8) on exit, we call 'del' explicitly
# Exception ignored in: <function Minio.__del__ at 0x15e67cca0>
del minioClient

# Now let's create the `tf.data` datasets:

AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

dataset = tf.data.TFRecordDataset(training_files, num_parallel_reads=AUTO)
dataset = dataset.with_options(ignore_order)

validation = tf.data.TFRecordDataset(validation_files, num_parallel_reads=AUTO)
validation = validation.with_options(ignore_order)

testing_dataset = tf.data.TFRecordDataset(testing_filenames, num_parallel_reads=AUTO)
testing_dataset = testing_dataset.with_options(ignore_order)


def decode_fn(record_bytes):
    schema = {
        "label": tf.io.FixedLenFeature([2], dtype=tf.int64),
        "sentence": tf.io.FixedLenFeature([512], dtype=tf.float32),
    }

    tf_example = tf.io.parse_single_example(record_bytes, schema)
    new_shape = tf.reshape(tf_example["sentence"], [1, 512])
    label = tf.reshape(tf_example["label"], [1, 2])
    return new_shape, label


model = keras.Sequential()

model.add(keras.layers.Dense(units=256, input_shape=(1, 512), activation="relu"))
model.add(keras.layers.Dropout(rate=0.5))

model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dropout(rate=0.5))

model.add(keras.layers.Dense(2, activation="softmax"))
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(0.001),
    metrics=["accuracy"],
)

# ![Structure of our Deep Learning model](pic2.png)


model.summary()

# Let's prepare our datasets for the training stage by batching 'BATCH_SIZE' items at a time
mapped_ds = dataset.map(decode_fn)
mapped_ds = mapped_ds.batch(BATCH_SIZE)

mapped_validation = validation.map(decode_fn)
mapped_validation = mapped_validation.batch(BATCH_SIZE)

testing_mapped_ds = testing_dataset.map(decode_fn)
testing_mapped_ds = testing_mapped_ds.batch(BATCH_SIZE)

checkpoint_path = f"s3://{DATASET_BUCKET}/checkpoints/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)

MODEL_NOTE = "256"
logdir = f"s3://{DATASET_BUCKET}/logs/imdb/{MODEL_NOTE}-" + datetime.now().strftime(
    "%Y%m%d-%H%M%S"
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Finally we will train the model:
history = model.fit(
    mapped_ds,
    epochs=10,
    callbacks=[cp_callback, tensorboard_callback],
    validation_data=mapped_validation,
)

# Now that we have our model, we want to save it to MinIO
# Note: This might raise a warning
#   (compare to: https://github.com/tensorflow/tensorboard/issues/48561)
model.save(f"s3://{DATASET_BUCKET}/imdb_sentiment_analysis")

samples = [
    "This movie sucks",
    "This was extremely good, I loved it.",
    "great acting",
    "terrible acting",
    "pure kahoot",
    "I don't know what's the point of this movie, this movie sucks but the acting is great",
    "This is not a good movie",
]

print("Here are some examples:")
sample_embedded = tf.expand_dims(embed(samples), axis=1)
res = model.predict(sample_embedded)
for i, sample in enumerate(samples):
    if res[i][0][0] > res[i][0][1]:
        print(f"\t{sample} - positive")
    else:
        print(f"\t{sample} - negative")

testing = model.evaluate(testing_mapped_ds)
print(f"Test set accuracy is at {testing[1]:.2f}")
