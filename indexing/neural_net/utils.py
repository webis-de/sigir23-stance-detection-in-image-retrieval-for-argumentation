import math
from pathlib import Path
from typing import List, Tuple

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

from indexing import FeatureIndex


def get_image(image_path, img_shape):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    resized = cv2.resize(img, (img_shape[0], img_shape[1]), interpolation=cv2.INTER_AREA)
    if len(img.shape) > 2 and img.shape[2] == 4:
        resized = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return resized


def split_data(data: pd.DataFrame, test: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = data.sample(frac=1)
    if not test:
        split_factor = 0.66
        df_len = len(data.index)
        split_index = round(df_len * split_factor)

        df_train = data.iloc[:split_index, :]
        df_test = data.iloc[split_index:, :]
    else:
        df_test = df.loc[df['topic'].isin(test)]
        df_train = df.loc[~df['topic'].isin(test)]
    return df_train, df_test


def get_text_position_data(data: pd.DataFrame) -> np.ndarray:
    tp_list = []
    for tp_str in data['text_position']:
        list_from_str = FeatureIndex.convert_text_area_from_str(tp_str)
        dim_heatmap = int(math.sqrt(len(list_from_str)))
        chunked_list = []
        for i in range(0, len(list_from_str), dim_heatmap):
            chunked_list.append(list_from_str[i:i + dim_heatmap])
        tp_list.append(chunked_list)
    return np.expand_dims(tp_list, axis=3)


def get_color_data(data: pd.DataFrame, cols_to_use: List[str] = None) -> np.ndarray:
    if cols_to_use is None or len(cols_to_use) == 0:
        cols = [
            'image_average_color_r',
            'image_average_color_g',
            'image_average_color_b',
            'image_dominant_color_r',
            'image_dominant_color_g',
            'image_dominant_color_b'
        ]
    else:
        cols = cols_to_use

    return np.asarray(data[cols])


def get_primary_arg_data(data: pd.DataFrame, cols_to_use: List[str] = None) -> np.ndarray:
    if cols_to_use is None or len(cols_to_use) == 0:
        cols = [
            'image_percentage_green',
            'image_percentage_red',
            'image_percentage_blue',
            'image_percentage_yellow',
            'image_percentage_bright',
            'image_percentage_dark',
            'text_len',
            'text_sentiment_score',
            'text_sentiment_score_con',
            'image_type',
            'image_roi_area'
        ]
    else:
        cols = cols_to_use

    return np.asarray(data[cols])


def get_primary_stance_data(data: pd.DataFrame, cols_to_use: List[str] = None) -> np.ndarray:
    if cols_to_use is None or len(cols_to_use) == 0:
        cols = [
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
            'query_sentiment',
            'query_sentiment_con',
            'query_html_eq',
            'query_image_eq',
            'query_html_context',
            'query_html_context_con',
            'query_image_context',
            'query_image_context_con',
            'query_image_align'
        ]
    else:
        cols = cols_to_use

    return np.asarray(data[cols])


def create_test_position_model(shape: Tuple[int, int, int]) -> keras.Model:
    model = Sequential([
        Conv2D(filters=1, kernel_size=2, input_shape=shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(6, activation='relu'),
        Dense(3, activation='relu')
    ])

    return model


def create_color_model(input_dim: int) -> keras.Model:
    if input_dim == 0:
        input_dim = 6
    color_model = Sequential()
    color_model.add(Dense(3, input_dim=input_dim, activation='relu'))
    return color_model


def plot_history(history, png_dir: Path):
    plt.set_loglevel('info')
    # summarize history for accuracy
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(png_dir.joinpath('/accuracy_function.png').as_posix())

    # summarize history for loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(png_dir.joinpath('/loss_function.png').as_posix())


def eval_to_just_one(data: List[int], one: str):
    output_data = []
    if one.lower() == 'pro':
        one_class = 1
    else:
        one_class = -1

    for value in data:
        if value == one_class:
            output_data.append(1)
        else:
            output_data.append(0)
    return np.asarray(output_data)


def get_pro_con_list(data: List[Tuple[float, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    pro_list = []
    con_list = []
    for value in data:
        pro_list.append(value[0])
        con_list.append(value[2])
    return np.array(pro_list), np.array(con_list)


def categorical_to_eval(data: List[Tuple[float, float, float]]) -> np.ndarray:
    output_data = []
    for value in data:
        if value[0] >= value[1] and value[0] > value[2]:
            output_data.append(1)
        elif value[2] >= value[0] and value[2] > value[1]:
            output_data.append(-1)
        else:
            output_data.append(0)

    return np.asarray(output_data)


def eval_to_categorical(data: List[int]) -> np.ndarray:
    output_data = []
    for value in data:
        if value == -1.0:
            output_data.append([0, 0, 1])
        elif value == 1.0:
            output_data.append([1, 0, 0])
        else:
            output_data.append([0, 1, 0])
    return np.asarray(output_data)
