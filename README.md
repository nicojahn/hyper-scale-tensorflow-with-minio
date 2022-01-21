# Hyper-Scale Machine Learning with MinIO and TensorFlow

This is an update of the code for the Blog Post: [Hyper-Scale Machine Learning with MinIO and TensorFlow](https://min.io)

This was built and tested using a macOS and Ubuntu system:
* Python 3.9
* TensorFlow 2.7.0
* TensorBoard 2.8.0 (>=2.8.0, as S3 support is added via TF IO: https://github.com/tensorflow/tensorboard/issues/5480)
* minio-py 7.1.2
* and all the depencies in the 'requirements.txt'

### How to use it
* Install depencies
```bash
python3 -m pip install -r requirements.txt
```

* Specifiy the necessary environment variables in the '.env' file
* Execute data pipeline, training and validation (Note: the data pipeline takes the longest)
```bash
python3 MinIO_Tensorflow.py
```

* Inspect the model via TensorBoard
 * Please note, that exporting your credentials might be a security risk (even if it is just temporarily), thus we unset it afterwards
```bash
export $(grep -v '^#' .env | xargs)
tensorboard --logdir s3://$DATASET_BUCKET/logs/
unset $(grep -v '^#' .env | cut -d'=' -f1 | xargs)
```

Copyright MinIO 2020
