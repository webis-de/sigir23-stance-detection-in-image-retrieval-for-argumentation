import abc
import gc
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Dict
import numpy as np

import keras
import keras.backend as K
import pandas as pd
from keras.models import Model
from keras import layers
from keras.applications import ResNet50V2
from keras.callbacks import EarlyStopping
from keras.models import load_model, Sequential

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa

from config import Config
from indexing.data_entry import DataEntry, Topic
from . import utils
from indexing.preprocessing import SpacyPreprocessor

# to get no console-print from tensorflow
from .. import FeatureIndex

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None
overfitCallback = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=8)
cfg = Config.get()
log = logging.getLogger('stance_network')


class StanceNetwork(abc.ABC):
    model: keras.Model
    name: str
    version: int
    dir_path: Path = cfg.working_dir.joinpath(Path('models/stance/'))

    def __init__(self, name: str, version: int):
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.version = version
        self.topics_to_skip = [15, 31, 36, 37, 43, 45, 48]
        self.cols_to_use = []

    @staticmethod
    def get(name: str, version: int = 3) -> 'StanceNetwork':
        if version == 1:
            return StanceNetworkV1(name)
        elif version == 2:
            return StanceNetworkV2(name)
        elif version == 3:
            return StanceNetworkV3(name)
        elif version == 4:
            return StanceNetworkV4(name)
        elif version == 5:
            return StanceNetworkV5(name)
        elif version == 6:
            return StanceNetworkV6(name)
        elif version == 7:
            return StanceNetworkV7(name)
        elif version == 8:
            return StanceNetworkV8(name)
        else:
            return StanceNetworkV9(name)

    @staticmethod
    def load(name: str, version: int = 2) -> 'StanceNetwork':
        arg_model = StanceNetwork.get(name, version)
        model_path = arg_model.dir_path.joinpath(name).joinpath('model.hS')
        if not model_path.exists():
            raise FileNotFoundError(f'The model {name} does not exists.')
        if version != 1:
            arg_model.reload()
        return arg_model

    def reload(self) -> None:
        model_path = self.dir_path.joinpath(self.name).joinpath('model.hS')
        if not model_path.exists():
            raise FileNotFoundError(f'The model {self.name} does not exists.')
        self.model = load_model(model_path.as_posix(), compile=False)

    def train(self, data: pd.DataFrame, test: List[int]) -> None:
        pass

    def predict(self, data: pd.DataFrame, **kwargs) -> List[float]:
        pass

    def set_cols_to_use(self, cols_to_use):
        self.cols_to_use = cols_to_use


class StanceNetworkV9(StanceNetwork):

    def __init__(self, name: str):
        super().__init__(name, 9)
        self.dir_path = self.dir_path.joinpath('version_9')
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.EPOCHS = 20
        self.BATCH_SIZE = 32

    def create_text_encoder(self, trainable=False):
        # Load the BERT preprocessing module.
        preprocess = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2",
            trainable=False,
        )
        # Load the pre-trained BERT model to be used as the base encoder.
        bert = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
            trainable=trainable,
        )
        # Set the trainability of the base encoder.
        bert.trainable = trainable
        # Receive the text as inputs.
        inputs = layers.Input(shape=(), dtype=tf.string)
        # Preprocess the text.
        bert_inputs = preprocess(inputs)
        # Generate embeddings for the preprocessed text using the BERT model.
        embeddings = bert(bert_inputs)["pooled_output"]
        # Project the embeddings produced by the model.
        x = tf.keras.layers.Dropout(0.5)(embeddings)

        # Create the text encoder model.
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def train(self, data: pd.DataFrame, test: List[int]) -> None:

        texts_image_html = []
        texts_query = []
        stance_encode = []
        fidx = FeatureIndex.load('23826')
        prep = SpacyPreprocessor()
        with fidx:
            for image_id, row in data.iterrows():

                ocr_text = fidx.get_image_text(str(image_id))
                ocr_text = ''.join(ocr_text)

                html_text = fidx.get_html_text(str(image_id))
                html_text = ''.join(html_text)

                query = prep.preprocess(Topic.get(row['topic']).title)
                query = ' '.join(query)

                texts_image_html.append(str(ocr_text + " " + html_text))
                texts_query.append(query)
                stance_encode.append(row['stance_eval'])

        index_to_split = math.ceil(len(texts_query) * 0.8)
        trainImages = np.array(texts_image_html[:index_to_split])
        testImages = np.array(texts_image_html[index_to_split:])
        trainTexts = np.array(texts_query[:index_to_split])
        testTexts = np.array(texts_query[index_to_split:])
        stance_encode_train = np.array(utils.eval_to_categorical(stance_encode[:index_to_split]))
        stance_encode_test = np.array(utils.eval_to_categorical(stance_encode[index_to_split:]))

        text_encoder_image = self.create_text_encoder()

        text_encoder_query = self.create_text_encoder()

        combinedInput = layers.concatenate([text_encoder_image.output, text_encoder_query.output])

        x3 = layers.Dense(128, activation="relu")(combinedInput)
        x3 = layers.Dropout(0.2)(x3)
        x2 = layers.Dense(32, activation="relu")(x3)
        x2 = layers.Dropout(0.2)(x2)
        x1 = layers.Dense(8, activation="relu")(x2)
        x1 = layers.Dropout(0.2)(x1)
        x = layers.Dense(3, activation="softmax")(x1)
        model = Model(inputs=[text_encoder_image.input, text_encoder_query.input], outputs=x)

        count_pro = 0
        count_neutral = 0
        count_con = 0
        count = 0
        for i in stance_encode_train:
            count += 1
            if i[0] == 1:
                count_con += 1
            if i[1] == 1:
                count_neutral += 1
            if i[2] == 1:
                count_pro += 1

        print("counts: ", count_con, count_neutral, count_pro)

        class_weight = {0: count_neutral / count_con,  # con 4.7
                        1: 1,  # neutral 1
                        2: count_neutral / count_pro}  # pro 3.3

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=["accuracy"])

        # Create a learning rate scheduler callback.
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3
        )

        # Create an early stopping callback.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = model.fit(x=[trainImages, trainTexts], y=stance_encode_train,
                            epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                            validation_data=([testImages, testTexts], stance_encode_test),
                            class_weight=class_weight,
                            callbacks=[reduce_lr, early_stopping])

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        log.info(f'Model {self.name} {self.version} history: {history.history}')

        return history.history['val_accuracy']

    def predict(self, data: pd.DataFrame, query: List[str] = None) -> List[int]:
        texts_image_html = []
        texts_query = []
        fidx = FeatureIndex.load('23826')
        # prep = SpacyPreprocessor()
        query = ' '.join(query)
        with fidx:
            for image_id, row in data.iterrows():
                ocr_text = fidx.get_image_text(str(image_id))
                ocr_text = ''.join(ocr_text)

                html_text = fidx.get_html_text(str(image_id))
                html_text = ''.join(html_text)

                texts_image_html.append(str(ocr_text + " " + html_text))
                texts_query.append(query)

        pred = self.model.predict(x=[np.array(texts_image_html), np.array(texts_query)])
        return utils.get_pro_con_list(pred)


class StanceNetworkV8(StanceNetwork):

    def __init__(self, name: str):
        super().__init__(name, 8)
        self.dir_path = self.dir_path.joinpath('version_8')
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.IMG_SHAPE = (244, 244, 3)
        self.EPOCHS = 30
        self.BATCH_SIZE = 32

    def project_embeddings(self, embeddings, num_projection_layers, projection_dims, dropout_rate):
        projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = layers.Dense(projection_dims)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Add()([projected_embeddings, x])
            projected_embeddings = layers.LayerNormalization()(x)
        return projected_embeddings

    def create_image_encoder(self, inputShape, trainable=False):
        # define resNet50 model
        base_cnn = ResNet50V2(
            weights="imagenet", input_shape=inputShape, include_top=False
        )
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = trainable
            layer.trainable = False

        inputs = keras.Input(shape=inputShape)

        data_augmentation = keras.Sequential(
            [keras.layers.RandomFlip("horizontal"),
             keras.layers.RandomRotation(0.1),
             ]
        )
        x = data_augmentation(inputs)

        x = tf.keras.applications.resnet_v2.preprocess_input(x)
        x = base_cnn(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        output = self.project_embeddings(x, 256, 1, 0.5)
        model = Model(inputs=inputs, outputs=output)

        # return the model to the calling function
        return model

    def create_text_encoder(self, trainable=False):
        # Load the BERT preprocessing module.
        preprocess = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2",
            trainable=False, name="text_preprocessing",
        )
        # Load the pre-trained BERT model to be used as the base encoder.
        bert = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
            trainable=trainable, name="bert",
        )
        # Set the trainability of the base encoder.
        bert.trainable = trainable
        # Receive the text as inputs.
        inputs = layers.Input(shape=(), dtype=tf.string, name="text_input")
        # Preprocess the text.
        bert_inputs = preprocess(inputs)
        # Generate embeddings for the preprocessed text using the BERT model.
        embeddings = bert(bert_inputs)["pooled_output"]
        # Project the embeddings produced by the model.
        output = self.project_embeddings(embeddings, 256, 1, 0.5)

        # Create the text encoder model.
        model = keras.Model(inputs=inputs, outputs=output, name="text_encoder")
        return model

    def train(self, data: pd.DataFrame, test: List[int]) -> None:

        images = []
        texts = []
        stance_encode = []
        fidx = FeatureIndex.load('23826')
        prep = SpacyPreprocessor()
        with fidx:
            for image_id, row in data.iterrows():
                image_path = DataEntry.load(image_id).png_path
                images.append(utils.get_image(image_path, self.IMG_SHAPE))

                ocr_text = fidx.get_image_text(str(image_id))
                ocr_text = ''.join(ocr_text)

                query = prep.preprocess(Topic.get(row['topic']).title)
                query = ' '.join(query)

                texts.append(str(query + " " + ocr_text))
                stance_encode.append(row['stance_eval'])

        index_to_split = math.ceil(len(texts) * 0.8)
        trainImages = np.array(images[:index_to_split])
        testImages = np.array(images[index_to_split:])
        trainTexts = np.array(texts[:index_to_split])
        testTexts = np.array(texts[index_to_split:])
        stance_encode_train = np.array(utils.eval_to_categorical(stance_encode[:index_to_split]))
        stance_encode_test = np.array(utils.eval_to_categorical(stance_encode[index_to_split:]))

        image_encoder = self.create_image_encoder(inputShape=self.IMG_SHAPE,
                                                  trainable=False)

        text_encoder = self.create_text_encoder()

        combinedInput = layers.concatenate([image_encoder.output, text_encoder.output])

        x3 = layers.Dense(128, activation="relu")(combinedInput)
        x3 = layers.Dropout(0.2)(x3)
        x2 = layers.Dense(32, activation="relu")(x3)
        x2 = layers.Dropout(0.2)(x2)
        x1 = layers.Dense(8, activation="relu")(x2)
        x1 = layers.Dropout(0.2)(x1)
        x = layers.Dense(3, activation="softmax")(x1)
        model = Model(inputs=[image_encoder.input, text_encoder.input], outputs=x)

        count_pro = 0
        count_neutral = 0
        count_con = 0
        count = 0
        for i in stance_encode_train:
            count += 1
            if i[0] == 1:
                count_con += 1
            if i[1] == 1:
                count_neutral += 1
            if i[2] == 1:
                count_pro += 1

        print("counts: ", count_con, count_neutral, count_pro)

        class_weight = {0: count_neutral / count_con,  # con 4.7
                        1: 1,  # neutral 1
                        2: count_neutral / count_pro}  # pro 3.3

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=0.00005),
                      metrics=["accuracy"])

        # Create a learning rate scheduler callback.
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3
        )

        # Create an early stopping callback.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = model.fit(x=[trainImages, trainTexts], y=stance_encode_train,
                            epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                            validation_data=([testImages, testTexts], stance_encode_test),
                            class_weight=class_weight,
                            callbacks=[reduce_lr, early_stopping])

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        log.info(f'Model {self.name} {self.version} history: {history.history}')

        return history.history['val_accuracy']

    def predict(self, data: pd.DataFrame, query: List[str] = None) -> List[int]:
        images = []
        texts = []
        fidx = FeatureIndex.load('23826')
        # prep = SpacyPreprocessor()
        query = ' '.join(query)
        with fidx:
            for image_id, row in data.iterrows():
                image_path = DataEntry.load(image_id).png_path
                images.append(utils.get_image(image_path, self.IMG_SHAPE))

                ocr_text = fidx.get_image_text(str(image_id))
                ocr_text = ''.join(ocr_text)

                # query = prep.preprocess(Topic.get(row['topic']).title)
                # query = ' '.join(query)

                texts.append(str(query + " " + ocr_text))

        pred = self.model.predict(x=[np.array(images), np.array(texts)])
        print(pred)
        return utils.get_pro_con_list(pred)


class StanceNetworkV7(StanceNetwork):

    def __init__(self, name: str, stance: str = 'PRO'):
        super().__init__(name, 7)
        self.dir_path = self.dir_path.joinpath('version_7')
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.IMG_SHAPE = (256, 256, 3)
        self.EPOCHS = 20
        self.BATCH_SIZE = 16
        self.stance = stance

    def project_embeddings(self, embeddings, num_projection_layers, projection_dims, dropout_rate):
        projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = layers.Dense(projection_dims)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Add()([projected_embeddings, x])
            projected_embeddings = layers.LayerNormalization()(x)
        return projected_embeddings

    def create_image_encoder(self, inputShape, num_projection_layers, projection_dims, dropout_rate, trainable=False):
        # define resNet50 model
        base_cnn = ResNet50V2(
            weights="imagenet", input_shape=inputShape, include_top=False
        )
        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        flatten = layers.Flatten()(base_cnn.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = self.project_embeddings(dense2, num_projection_layers, projection_dims, dropout_rate)

        # build the model
        model = Model(inputs=base_cnn.input, outputs=output, name="vision_encoder")

        # return the model to the calling function
        return model

    def create_text_encoder(self, num_projection_layers, projection_dims, dropout_rate, trainable=False):
        # Load the BERT preprocessing module.
        preprocess = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2",
            trainable=False, name="text_preprocessing",
        )
        # Load the pre-trained BERT model to be used as the base encoder.
        bert = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
            trainable=False, name="bert",
        )
        # Set the trainability of the base encoder.
        bert.trainable = trainable
        # Receive the text as inputs.
        inputs = layers.Input(shape=(), dtype=tf.string, name="text_input")
        # Preprocess the text.
        bert_inputs = preprocess(inputs)
        # Generate embeddings for the preprocessed text using the BERT model.
        embeddings = bert(bert_inputs)["pooled_output"]
        # Project the embeddings produced by the model.
        outputs = self.project_embeddings(
            embeddings, num_projection_layers, projection_dims, dropout_rate
        )
        # Create the text encoder model.
        model = keras.Model(inputs, outputs, name="text_encoder")
        return model

    def train(self, data: pd.DataFrame, test: List[int]) -> None:

        images = []
        texts = []
        stance_encode = []
        fidx = FeatureIndex.load('23826')
        prep = SpacyPreprocessor()
        with fidx:
            for image_id, row in data.iterrows():
                image_path = DataEntry.load(image_id).png_path
                images.append(utils.get_image(image_path, self.IMG_SHAPE))

                ocr_text = fidx.get_image_text(str(image_id))
                ocr_text = ''.join(ocr_text)

                query = prep.preprocess(Topic.get(row['topic']).title)
                query = ' '.join(query)

                texts.append(str(query + " " + ocr_text))
                stance_encode.append(row['stance_eval'])

        index_to_split = math.ceil(len(texts) * 0.8)
        trainImages = np.array(images[:index_to_split])
        testImages = np.array(images[index_to_split:])
        trainTexts = np.array(texts[:index_to_split])
        testTexts = np.array(texts[index_to_split:])
        stance_encode_train = np.array(utils.eval_to_just_one(stance_encode[:index_to_split], one=self.stance))
        stance_encode_test = np.array(utils.eval_to_just_one(stance_encode[index_to_split:], one=self.stance))

        weight = len(stance_encode_train) / np.count_nonzero(stance_encode_train == 0)
        class_weight = {0: 1,
                        1: weight}

        image_encoder = self.create_image_encoder(inputShape=self.IMG_SHAPE,
                                                  num_projection_layers=1,
                                                  projection_dims=256,
                                                  dropout_rate=0.1)

        text_encoder = self.create_text_encoder(num_projection_layers=1,
                                                projection_dims=256,
                                                dropout_rate=0.1)

        combinedInput = layers.concatenate([image_encoder.output, text_encoder.output])

        x3 = layers.Dense(128, activation="relu")(combinedInput)
        x2 = layers.Dense(32, activation="relu")(x3)
        x1 = layers.Dense(8, activation="relu")(x2)
        x = layers.Dense(1, activation="sigmoid")(x1)
        model = Model(inputs=[image_encoder.input, text_encoder.input], outputs=x)

        model.compile(loss='binary_crossentropy',
                      optimizer=tfa.optimizers.AdamW(learning_rate=0.00005, weight_decay=0.00005),
                      metrics=["accuracy"])

        # Create a learning rate scheduler callback.
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3
        )

        # Create an early stopping callback.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = model.fit(x=[trainImages, trainTexts], y=stance_encode_train,
                            epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                            class_weight=class_weight,
                            validation_data=([testImages, testTexts], stance_encode_test),
                            callbacks=[reduce_lr, early_stopping])

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        log.info(f'Model {self.name} {self.version} history: {history.history}')

        return history.history['val_accuracy']

    def predict(self, data: pd.DataFrame, query: List[str] = None) -> List[int]:
        images = []
        texts = []
        fidx = FeatureIndex.load('23826')
        # prep = SpacyPreprocessor()
        query = ' '.join(query)
        with fidx:
            for image_id, row in data.iterrows():
                image_path = DataEntry.load(image_id).png_path
                images.append(utils.get_image(image_path, self.IMG_SHAPE))

                ocr_text = fidx.get_image_text(str(image_id))
                ocr_text = ''.join(ocr_text)

                # query = prep.preprocess(Topic.get(row['topic']).title)
                # query = ' '.join(query)

                texts.append(str(query + " " + ocr_text))

        # self.reload()
        pred = self.model.predict(x=[np.array(images), np.array(texts)], batch_size=self.BATCH_SIZE)
        # del self.model
        # gc.collect()
        # tf.keras.backend.clear_session()
        return pred


class StanceNetworkV6(StanceNetwork):

    def __init__(self, name: str):
        super().__init__(name, 6)
        self.dir_path = self.dir_path.joinpath('version_6')
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.IMG_SHAPE = (256, 256, 3)
        self.EPOCHS = 20
        self.BATCH_SIZE = 32

    def project_embeddings(self, embeddings, num_projection_layers, projection_dims, dropout_rate):
        projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = layers.Dense(projection_dims)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Add()([projected_embeddings, x])
            projected_embeddings = layers.LayerNormalization()(x)
        return projected_embeddings

    def create_image_encoder(self, inputShape, num_projection_layers, projection_dims, dropout_rate, trainable=False):
        # define resNet50 model
        base_cnn = ResNet50V2(
            weights="imagenet", input_shape=inputShape, include_top=False
        )
        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        flatten = layers.Flatten()(base_cnn.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = self.project_embeddings(dense2, num_projection_layers, projection_dims, dropout_rate)

        # build the model
        model = Model(inputs=base_cnn.input, outputs=output, name="vision_encoder")

        # return the model to the calling function
        return model

    def create_text_encoder(self, num_projection_layers, projection_dims, dropout_rate, trainable=False):
        # Load the BERT preprocessing module.
        preprocess = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2",
            trainable=False, name="text_preprocessing",
        )
        # Load the pre-trained BERT model to be used as the base encoder.
        bert = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
            trainable=False, name="bert",
        )
        # Set the trainability of the base encoder.
        bert.trainable = trainable
        # Receive the text as inputs.
        inputs = layers.Input(shape=(), dtype=tf.string, name="text_input")
        # Preprocess the text.
        bert_inputs = preprocess(inputs)
        # Generate embeddings for the preprocessed text using the BERT model.
        embeddings = bert(bert_inputs)["pooled_output"]
        # Project the embeddings produced by the model.
        outputs = self.project_embeddings(
            embeddings, num_projection_layers, projection_dims, dropout_rate
        )
        # Create the text encoder model.
        model = keras.Model(inputs, outputs, name="text_encoder")
        return model

    def train(self, data: pd.DataFrame, test: List[int]) -> None:

        images = []
        texts = []
        stance_encode = []
        fidx = FeatureIndex.load('23826')
        prep = SpacyPreprocessor()
        with fidx:
            for image_id, row in data.iterrows():
                image_path = DataEntry.load(image_id).png_path
                images.append(utils.get_image(image_path, self.IMG_SHAPE))

                ocr_text = fidx.get_image_text(str(image_id))
                ocr_text = ''.join(ocr_text)

                query = prep.preprocess(Topic.get(row['topic']).title)
                query = ' '.join(query)

                texts.append(str(query + " " + ocr_text))
                stance_encode.append(row['stance_eval'])

        index_to_split = math.ceil(len(texts) * 0.8)
        trainImages = np.array(images[:index_to_split])
        testImages = np.array(images[index_to_split:])
        trainTexts = np.array(texts[:index_to_split])
        testTexts = np.array(texts[index_to_split:])
        stance_encode_train = np.array(utils.eval_to_categorical(stance_encode[:index_to_split]))
        stance_encode_test = np.array(utils.eval_to_categorical(stance_encode[index_to_split:]))

        image_encoder = self.create_image_encoder(inputShape=self.IMG_SHAPE,
                                                  num_projection_layers=1,
                                                  projection_dims=256,
                                                  dropout_rate=0.1)

        text_encoder = self.create_text_encoder(num_projection_layers=1,
                                                projection_dims=256,
                                                dropout_rate=0.1)

        combinedInput = layers.concatenate([image_encoder.output, text_encoder.output])

        x3 = layers.Dense(128, activation="relu")(combinedInput)
        x2 = layers.Dense(32, activation="relu")(x3)
        x1 = layers.Dense(8, activation="relu")(x2)
        x = layers.Dense(3, activation="sigmoid")(x1)
        model = Model(inputs=[image_encoder.input, text_encoder.input], outputs=x)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tfa.optimizers.AdamW(learning_rate=0.00005, weight_decay=0.00005),
                      metrics=["accuracy"])

        # Create a learning rate scheduler callback.
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3
        )

        # Create an early stopping callback.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = model.fit(x=[trainImages, trainTexts], y=stance_encode_train,
                            epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                            validation_data=([testImages, testTexts], stance_encode_test),
                            callbacks=[reduce_lr, early_stopping])

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        log.info(f'Model {self.name} {self.version} history: {history.history}')

        return history.history['val_accuracy']

    def predict(self, data: pd.DataFrame, query: List[str] = None) -> List[int]:
        images = []
        texts = []
        fidx = FeatureIndex.load('23826')
        # prep = SpacyPreprocessor()
        query = ' '.join(query)
        with fidx:
            for image_id, row in data.iterrows():
                image_path = DataEntry.load(image_id).png_path
                images.append(utils.get_image(image_path, self.IMG_SHAPE))

                ocr_text = fidx.get_image_text(str(image_id))
                ocr_text = ''.join(ocr_text)

                # query = prep.preprocess(Topic.get(row['topic']).title)
                # query = ' '.join(query)

                texts.append(str(query + " " + ocr_text))

        pred = self.model.predict(x=[np.array(images), np.array(texts)])
        return utils.get_pro_con_list(pred)


class StanceNetworkV5(StanceNetwork):

    def __init__(self, name: str):
        super().__init__(name, 5)
        self.dir_path = self.dir_path.joinpath('version_5')
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.cols_to_use = [
            'html_sentiment_score',
            'html_sentiment_score_con',
            'text_len',
            'text_sentiment_score',
            'text_sentiment_score_con',
            'query_html_eq',
            'query_image_eq',
            'query_html_context',
            'query_html_context_con',
            'query_image_context',
            'query_image_context_con',
            'query_image_align'
        ]
        self.IMG_SHAPE = (256, 256, 3)
        self.EPOCHS = 20
        self.BATCH_SIZE = 32

    def project_embeddings(self, embeddings, num_projection_layers, projection_dims, dropout_rate):
        projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = layers.Dense(projection_dims)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Add()([projected_embeddings, x])
            projected_embeddings = layers.LayerNormalization()(x)
        return projected_embeddings

    def create_feature_encoder(self, dim=12):
        # define our MLP network
        model = Sequential()
        model.add(layers.Dense(8, input_dim=dim, activation="relu"))
        model.add(layers.Dense(4, activation="relu"))
        return model

    def create_image_encoder(self, inputShape, num_projection_layers, projection_dims, dropout_rate, trainable=False):
        # define resNet50 model
        base_cnn = ResNet50V2(
            weights="imagenet", input_shape=inputShape, include_top=False
        )
        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        flatten = layers.Flatten()(base_cnn.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = self.project_embeddings(dense2, num_projection_layers, projection_dims, dropout_rate)

        # build the model
        model = Model(inputs=base_cnn.input, outputs=output, name="vision_encoder")

        # return the model to the calling function
        return model

    def create_text_encoder(self, num_projection_layers, projection_dims, dropout_rate, trainable=False):
        # Load the BERT preprocessing module.
        preprocess = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2",
            trainable=False, name="text_preprocessing",
        )
        # Load the pre-trained BERT model to be used as the base encoder.
        bert = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
            trainable=False, name="bert",
        )
        # Set the trainability of the base encoder.
        bert.trainable = trainable
        # Receive the text as inputs.
        inputs = layers.Input(shape=(), dtype=tf.string, name="text_input")
        # Preprocess the text.
        bert_inputs = preprocess(inputs)
        # Generate embeddings for the preprocessed text using the BERT model.
        embeddings = bert(bert_inputs)["pooled_output"]
        # Project the embeddings produced by the model.
        outputs = self.project_embeddings(
            embeddings, num_projection_layers, projection_dims, dropout_rate
        )
        # Create the text encoder model.
        model = keras.Model(inputs, outputs, name="text_encoder")
        return model

    def train(self, data: pd.DataFrame, test: List[int]) -> None:

        images = []
        texts = []
        features = []
        stance_encode = []
        fidx = FeatureIndex.load('23826')
        prep = SpacyPreprocessor()
        with fidx:
            for image_id, row in data.iterrows():
                image_path = DataEntry.load(image_id).png_path
                images.append(utils.get_image(image_path, self.IMG_SHAPE))

                ocr_text = fidx.get_image_text(str(image_id))
                ocr_text = ''.join(ocr_text)

                query = prep.preprocess(Topic.get(row['topic']).title)

                texts.append(ocr_text)
                feature = row[self.cols_to_use].values.flatten().tolist()
                features.append(feature)
                stance_encode.append(row['stance_eval'])

        index_to_split = math.ceil(len(texts) * 0.8)
        trainImages = np.array(images[:index_to_split])
        testImages = np.array(images[index_to_split:])
        trainTexts = np.array(texts[:index_to_split])
        testTexts = np.array(texts[index_to_split:])
        trainFeatures = np.array(features[:index_to_split])
        testFeatures = np.array(features[index_to_split:])
        stance_encode_train = np.array(utils.eval_to_categorical(stance_encode[:index_to_split]))
        stance_encode_test = np.array(utils.eval_to_categorical(stance_encode[index_to_split:]))

        feature_encoder = self.create_feature_encoder(dim=len(self.cols_to_use))

        image_encoder = self.create_image_encoder(inputShape=self.IMG_SHAPE,
                                                  num_projection_layers=1,
                                                  projection_dims=256,
                                                  dropout_rate=0.1)

        text_encoder = self.create_text_encoder(num_projection_layers=1,
                                                projection_dims=256,
                                                dropout_rate=0.1)

        combinedInput = layers.concatenate([image_encoder.output, text_encoder.output, feature_encoder.output])

        x3 = layers.Dense(128, activation="relu")(combinedInput)
        x2 = layers.Dense(32, activation="relu")(x3)
        x1 = layers.Dense(8, activation="relu")(x2)
        x = layers.Dense(3, activation="sigmoid")(x1)
        model = Model(inputs=[image_encoder.input, text_encoder.input, feature_encoder.input], outputs=x)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tfa.optimizers.AdamW(learning_rate=0.00005, weight_decay=0.00005),
                      metrics=["accuracy"])

        # Create a learning rate scheduler callback.
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3
        )

        # Create an early stopping callback.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = model.fit(x=[trainImages, trainTexts, trainFeatures], y=stance_encode_train,
                            epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                            validation_data=([testImages, testTexts, testFeatures], stance_encode_test),
                            callbacks=[reduce_lr, early_stopping])

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        log.info(f'Model {self.name} history: {history.history}')

        return history.history['val_accuracy']

    def predict(self, data: pd.DataFrame, query: List[str] = None) -> List[int]:
        images = []
        texts = []
        features = []
        fidx = FeatureIndex.load('23826')
        with fidx:
            for image_id, row in data.iterrows():
                image_path = DataEntry.load(image_id).png_path
                images.append(utils.get_image(image_path, self.IMG_SHAPE))

                ocr_text = fidx.get_image_text(str(image_id))
                ocr_text = ''.join(ocr_text)

                texts.append(ocr_text)
                feature = row[self.cols_to_use].values.flatten().tolist()
                features.append(feature)

        predictions = self.model.predict(x=[np.array(images), np.array(texts), np.array(features)])
        return utils.get_pro_con_list(predictions)


class StanceNetworkV4(StanceNetwork):

    def __init__(self, name: str):
        super().__init__(name, 4)
        self.dir_path = self.dir_path.joinpath('version_4')
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.IMG_SHAPE = (256, 256, 3)
        self.EPOCHS = 20
        self.BATCH_SIZE = 4

    def make_pairs(self, images, texts, labels, flag):
        # initialize two empty lists to hold the (image, image) pairs and
        # labels to indicate if a pair is positive or negative
        pairInput_image = []
        pairInput_text = []
        pairLabels = []

        # loop over all images
        for idxA, (currentImage, currentText, label) in enumerate(zip(images, texts, labels)):
            # grab the current image and label belonging to the current iteration
            # randomly pick an image that belongs to the *same* class label
            idxB = [i for i, x in enumerate(labels) if x == label]
            if idxA in idxB:
                idxB.remove(idxA)
            random_i = random.randint(0, len(idxB) - 1)
            posImage = images[idxB[random_i]]
            posText = texts[idxB[random_i]]
            # prepare a positive pair and update the images and labels lists, respectively
            pairInput_image.append([currentImage, posImage])
            pairInput_text.append([currentText, posText])
            pairLabels.append(1)
            # grab the indices for each of the class labels *not* equal to the current label and randomly pick an
            # image corresponding to a label *not* equal to the current label
            negIdx = [i for i, x in enumerate(labels) if x != label]
            random_i = random.randint(0, len(negIdx) - 1)
            negImage = images[negIdx[random_i]]
            negText = texts[negIdx[random_i]]
            # prepare a negative pair of images and update our lists
            pairInput_image.append([currentImage, negImage])
            pairInput_text.append([currentText, negText])
            pairLabels.append(0)
        # return a 2-tuple of our image pairs and labels

        split_index = math.ceil(len(pairLabels) * 0.8)
        if flag == "train":
            return (np.stack(pairInput_image[:split_index], axis=0),
                    np.stack(pairInput_text[:split_index], axis=0),
                    np.stack(pairLabels[:split_index], axis=0))
        else:
            return (np.stack(pairInput_image[split_index:], axis=0),
                    np.stack(pairInput_text[split_index:], axis=0),
                    np.stack(pairLabels[split_index:], axis=0))

    def euclidean_distance(self, vectors):
        # unpack the vectors into separate lists
        (featsA, featsB) = vectors
        # compute the sum of squared distances between the vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                           keepdims=True)
        # return the euclidean distance between the vectors
        return K.sqrt(K.maximum(sumSquared, K.epsilon()))

    def project_embeddings(self, embeddings, num_projection_layers, projection_dims, dropout_rate):
        projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = layers.Dense(projection_dims)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Add()([projected_embeddings, x])
            projected_embeddings = layers.LayerNormalization()(x)
        return projected_embeddings

    def create_image_encoder(self, inputShape, num_projection_layers, projection_dims, dropout_rate, trainable=False):
        # define resNet50 model
        base_cnn = ResNet50V2(
            weights="imagenet", input_shape=inputShape, include_top=False
        )
        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        flatten = layers.Flatten()(base_cnn.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = self.project_embeddings(dense2, num_projection_layers, projection_dims, dropout_rate)

        # build the model
        model = Model(inputs=base_cnn.input, outputs=output, name="vision_encoder")

        # return the model to the calling function
        return model

    def create_text_encoder(self, num_projection_layers, projection_dims, dropout_rate, trainable=False):
        # Load the BERT preprocessing module.
        preprocess = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2",
            trainable=False, name="text_preprocessing",
        )
        # Load the pre-trained BERT model to be used as the base encoder.
        bert = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
            trainable=False, name="bert",
        )
        # Set the trainability of the base encoder.
        bert.trainable = trainable
        # Receive the text as inputs.
        inputs = layers.Input(shape=(), dtype=tf.string, name="text_input")
        # Preprocess the text.
        bert_inputs = preprocess(inputs)
        # Generate embeddings for the preprocessed text using the BERT model.
        embeddings = bert(bert_inputs)["pooled_output"]
        # Project the embeddings produced by the model.
        outputs = self.project_embeddings(
            embeddings, num_projection_layers, projection_dims, dropout_rate
        )
        # Create the text encoder model.
        model = keras.Model(inputs, outputs, name="text_encoder")
        return model

    def build_siamese_model(self):
        image_encoder = self.create_image_encoder(inputShape=self.IMG_SHAPE,
                                                  num_projection_layers=1,
                                                  projection_dims=256,
                                                  dropout_rate=0.1)

        text_encoder = self.create_text_encoder(num_projection_layers=1,
                                                projection_dims=256,
                                                dropout_rate=0.1)

        combined = layers.concatenate([image_encoder.output, text_encoder.output])

        '''
        combinedInput = layers.concatenate([image_encoder.output, text_encoder.output, feature_encoder.output])
        
        x3 = layers.Dense(128, activation="relu")(combinedInput)
        x2 = layers.Dense(32, activation="relu")(x3)
        x1 = layers.Dense(8, activation="relu")(x2)
        x = layers.Dense(3, activation="sigmoid")(x1)
        model = Model(inputs=[image_encoder.input, text_encoder.input, feature_encoder.input], outputs=x)
        '''

        single_model = Model(inputs=[image_encoder.input, text_encoder.input], outputs=combined)
        input_layer_1 = [layers.Input(shape=self.IMG_SHAPE),
                         layers.Input(shape=(), dtype=tf.string, name="text_input_1")]
        input_layer_2 = [layers.Input(shape=self.IMG_SHAPE),
                         layers.Input(shape=(), dtype=tf.string, name="text_input_2")]

        model_1 = single_model(input_layer_1)
        model_2 = single_model(input_layer_2)

        # distance = layers.Lambda(self.euclidean_distance)([model_1, model_2])
        combined_end = layers.concatenate([model_1, model_2])
        x3 = layers.Dense(256, activation="relu")(combined_end)
        x2 = layers.Dense(128, activation="relu")(x3)
        x = layers.Dense(32, activation="relu")(x2)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = Model(inputs=[input_layer_1[0], input_layer_1[1], input_layer_2[0], input_layer_2[1]], outputs=outputs)

        return model

    def train(self, data: pd.DataFrame, test: List[int]) -> None:
        images = []
        texts = []
        fidx = FeatureIndex.load('23826')
        with fidx:
            for image_id, row in data.iterrows():
                image_path = DataEntry.load(image_id).png_path
                images.append(utils.get_image(image_path, self.IMG_SHAPE))

                ocr_text = fidx.get_image_text(str(image_id))
                ocr_text = ''.join(ocr_text)

                texts.append(ocr_text)

        labels = data['stance_eval']

        (pairTrain_image, pairTrain_text, labelTrain) = self.make_pairs(images, texts, labels, flag="train")
        (pairTest_image, pairTest_text, labelTest) = self.make_pairs(images, texts, labels, flag="test")

        model = self.build_siamese_model()

        print(model.input_shape)
        print(model.summary())

        # Create an early stopping callback.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        # Create a learning rate scheduler callback.
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5
        )

        model.compile(loss='binary  _crossentropy',
                      optimizer=tfa.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.0001),
                      metrics=["accuracy"])

        history = model.fit(
            x=[pairTrain_image[:, 0], pairTrain_text[:, 0], pairTrain_image[:, 1], pairTrain_text[:, 1]],
            y=labelTrain,
            validation_data=([[pairTest_image[:, 0], pairTest_text[:, 0]], [pairTest_image[:, 1], pairTest_text[:, 1]]],
                             labelTest),
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            callbacks=[early_stopping, reduce_lr])

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        utils.plot_history(history, self.dir_path.joinpath(self.name))

        return history.history['val_accuracy']

    def predict(self, data: pd.DataFrame, **kwargs) -> List[int]:
        images = []
        texts = []
        fidx = FeatureIndex.load('23826')
        with fidx:
            for image_id, row in data.iterrows():
                image_path = DataEntry.load(image_id).png_path
                images.append(utils.get_image(image_path, self.IMG_SHAPE))

                ocr_text = fidx.get_image_text(str(image_id))
                ocr_text = ''.join(ocr_text)

                texts.append(ocr_text)

        HOW_MUCH = 100
        labels = data['stance_eval'].to_list()[:HOW_MUCH]
        id_list = data.index.values[:HOW_MUCH]
        print(id_list)
        pos_list = list(range(HOW_MUCH))

        distance_matrix = np.zeros(shape=(HOW_MUCH, HOW_MUCH))

        for pos in pos_list[:-1]:
            pos_pairs = [(pos, b) for b in pos_list[pos + 1:]]

            id_pairs = []
            pair_image = []
            pair_text = []
            for pos1, pos2 in pos_pairs:
                id_pairs.append([id_list[pos1], id_list[pos2]])
                pair_image.append([images[pos1], images[pos2]])
                pair_text.append([texts[pos1], texts[pos2]])

            pair_image = np.stack(pair_image, axis=0)
            pair_text = np.stack(pair_text, axis=0)

            predictions = self.model.predict(x=[pair_image[:, 0], pair_text[:, 0], pair_image[:, 1], pair_text[:, 1]])

            print(pos, " : ", predictions)
            for i, (pos1, pos2) in enumerate(pos_pairs):
                distance_matrix[pos1, pos2] = predictions[i]

            np.savetxt("distance_matrix.txt", distance_matrix, fmt='%1.6f')

        for row in range(100):
            for column in range(row):
                distance_matrix[row, column] = distance_matrix[column, row]
        np.savetxt("distance_matrix.txt", distance_matrix, fmt='%1.6f')

        return id_list, distance_matrix, labels


class StanceNetworkV3(StanceNetwork):
    """
    Using Class-Weights
    """

    def __init__(self, name: str):
        super().__init__(name, 3)
        self.dir_path = self.dir_path.joinpath('version_3')
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.cols_to_use = [
            'image_percentage_green',
            'image_percentage_red',
            'image_percentage_blue',
            'image_percentage_yellow',
            'image_percentage_bright',
            'image_percentage_dark',
            'html_sentiment_score',
            'html_sentiment_score_con',
            'text_len',
            'text_sentiment_score',
            'text_sentiment_score_con',
            'image_average_color_r',
            'image_average_color_g',
            'image_average_color_b',
            'image_dominant_color_r',
            'image_dominant_color_g',
            'image_dominant_color_b',
            'query_html_eq',
            'query_image_eq',
            'query_html_context',
            'query_html_context_con',
            'query_image_context',
            'query_image_context_con',
            'query_image_align'
        ]

    def train(self, data: pd.DataFrame, test: List[int]) -> None:
        df_train, df_test = utils.split_data(data, test)
        y_train = utils.eval_to_categorical(df_train['stance_eval'].to_list())
        y_test = utils.eval_to_categorical(df_test['stance_eval'].to_list())

        primary_in_train = utils.get_primary_stance_data(df_train, cols_to_use=self.cols_to_use)
        primary_in_test = utils.get_primary_stance_data(df_test, cols_to_use=self.cols_to_use)

        model = Sequential([
            layers.Dense(40, input_dim=primary_in_train.shape[1], activation='relu'),
            layers.Dense(20, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])

        count_pro = 0
        count_neutral = 0
        count_con = 0
        count = 0
        for i in y_train:
            count += 1
            if i[0] == 1:
                count_con += 1
            if i[1] == 1:
                count_neutral += 1
            if i[2] == 1:
                count_pro += 1

        class_weight = {0: count_neutral / count_con,  # con 4.7
                        1: 1,  # neutral 1
                        2: count_neutral / count_pro}  # pro 3.3

        log.debug('Class weight for train %s', class_weight)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

        history = model.fit(x=primary_in_train, y=y_train,
                            epochs=120, batch_size=18,
                            validation_data=(primary_in_test, y_test),
                            class_weight=class_weight, verbose=0,
                            callbacks=[overfitCallback])

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        log.info(f'Model {self.name} history: {history.history}')
        # utils.plot_history(history, self.dir_path.joinpath(self.name))

        return history.history['val_accuracy']

    def predict(self, data: pd.DataFrame, **kwargs) -> List[int]:
        # tp_in = get_text_position_data(data)
        # color_in = get_color_data(data)
        primary_in = utils.get_primary_stance_data(data, cols_to_use=self.cols_to_use)

        predictions = self.model.predict(x=primary_in)
        return utils.get_pro_con_list(predictions)


class StanceNetworkV2(StanceNetwork):
    """
    New Features
    QueryInformation- and HTML-TextInformation-Usage
    """

    def __init__(self, name: str):
        super().__init__(name, 2)
        self.dir_path = self.dir_path.joinpath('version_2')
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.cols_to_get = []

    def train(self, data: pd.DataFrame, test: List[int]) -> None or Dict:
        df_train, df_test = utils.split_data(data, test)
        y_train = utils.eval_to_categorical(df_train['stance_eval'].to_list())
        y_test = utils.eval_to_categorical(df_test['stance_eval'].to_list())

        primary_in_train = utils.get_primary_stance_data(df_train)
        primary_in_test = utils.get_primary_stance_data(df_test)

        model = Sequential([
            layers.Dense(20, input_dim=primary_in_train.shape[1], activation='relu'),
            layers.Dense(15, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

        history = model.fit(x=primary_in_train, y=y_train,
                            epochs=100, batch_size=36,
                            validation_data=(primary_in_test, y_test))
        # callbacks=[overfitCallback])

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        utils.plot_history(history, self.dir_path.joinpath(self.name))

    def predict(self, data: pd.DataFrame, **kwargs) -> List[int]:
        # tp_in = get_text_position_data(data)
        # color_in = get_color_data(data)
        primary_in = utils.get_primary_stance_data(data)

        predictions = self.model.predict(x=primary_in)
        return utils.get_pro_con_list(predictions)


class StanceNetworkV1(StanceNetwork):
    """
    Model with just same features as the Argument-Model
    No queryInformation-usage
    """

    def __init__(self, name: str):
        super().__init__(name, 1)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.cols_to_get_primary = [
            'image_percentage_green',
            'image_percentage_red',
            'image_percentage_bright',
            'image_percentage_dark',
            'html_sentiment_score',
            'html_sentiment_score_con',
            'text_len',
            'text_sentiment_score',
            'text_sentiment_score_con',
            'image_average_color_r',
            'image_average_color_g',
            'image_average_color_b',
        ]

    def train(self, data: pd.DataFrame, test: List[int]) -> None:
        df_train, df_test = utils.split_data(data, test)
        y_train = utils.eval_to_categorical(df_train['stance_eval'].to_list())
        y_test = utils.eval_to_categorical(df_test['stance_eval'].to_list())

        primary_in_train = utils.get_primary_stance_data(df_train, cols_to_use=self.cols_to_get_primary)
        primary_in_test = utils.get_primary_stance_data(df_test, cols_to_use=self.cols_to_get_primary)

        model = Sequential([
            layers.Dense(15, input_dim=primary_in_train.shape[1], activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

        history = model.fit(x=primary_in_train, y=y_train,
                            epochs=100, batch_size=36,
                            validation_data=(primary_in_test, y_test))
        # callbacks=[overfitCallback])

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        utils.plot_history(history, self.dir_path.joinpath(self.name))

    def predict(self, data: pd.DataFrame, **kwargs) -> List[int]:
        primary_in = utils.get_primary_stance_data(data, cols_to_use=self.cols_to_get_primary)

        predictions = self.model.predict(x=primary_in)
        return utils.get_pro_con_list(predictions)
