import math
from typing import List, Union, Tuple, Any, Literal

import pandas as pd
from tqdm import tqdm

from config import Config
from evaluation.eval_data import get_df
from indexing import Topic, FeatureIndex, SpacyPreprocessor, DataEntry
from .similarity_funcs import alignment_query, context_sentiment, query_frequency
from ..feature import sentiment_detection
from . import utils
from evaluation.analysis_labeled_data import qrels_to_labeled_data

cfg = Config.get()


def preprocessed_data(fidx: FeatureIndex, topics: List[Topic], train_on: Literal['aramis', 'touche-qrels'] = 'aramis',
                      train: bool = False, with_query: bool = True) -> pd.DataFrame:
    data = pd.DataFrame()
    prep = SpacyPreprocessor()
    for topic in tqdm(topics, desc='Preprocess topics:'):
        query = prep.preprocess(topic.title) if with_query else []
        df = preprocess_data(fidx, topic.get_image_ids(), query,
                             train=train, topic=topic, train_on=train_on)
        data = pd.concat([data, df], axis=0)
    return data


def preprocess_data(fidx: FeatureIndex, ids: List[str], query: List[str], train: bool = False,
                    topic: Topic = None, train_on: Literal['aramis', 'touche-qrels'] = 'aramis',) -> pd.DataFrame:

    new_ids = [img_id for img_id in ids if img_id in fidx.get_image_ids()]
    data: pd.DataFrame = fidx.dataframe.loc[new_ids, :].copy()

    if train:
        if topic is None:
            raise ValueError('No topic defined for trained data preprocess')

        if train_on == 'aramis':
            t_df: pd.DataFrame = get_df().loc[(slice(None), slice(None), topic.number), :]
        elif train_on == 'touche-qrels':
            t_df: pd.DataFrame = qrels_to_labeled_data().loc[(slice(None), slice(None), topic.number), :]
        else:
            raise ValueError(f'Parameter train_on must be ether "aramis" or "touche-qrels", not {train_on}')
        t_df = t_df.loc[t_df['Topic_correct'], :]

        faulty_ids = t_df.index.unique(0).tolist()
        valid_ids = data.index.unique(0).tolist()

        correct_ids = [image_id for image_id in faulty_ids if image_id in valid_ids]
        t_df = t_df.loc[correct_ids, :]

        data.loc[t_df.index.unique(0), 'topic'] = topic.number
        data.loc[t_df.loc[(t_df['Argumentative'] == 'STRONG'), :].index.unique(0), 'arg_eval'] = 1
        data.loc[t_df.loc[(t_df['Argumentative'] == 'WEAK'), :].index.unique(0), 'arg_eval'] = 0.5
        data.loc[t_df.loc[(t_df['Argumentative'] == 'NONE'), :].index.unique(0), 'arg_eval'] = 0

        data.loc[t_df.loc[(t_df['Stance'] == 'PRO'), :].index.unique(0), 'stance_eval'] = 1
        data.loc[t_df.loc[(t_df['Stance'] == 'NEUTRAL'), :].index.unique(0), 'stance_eval'] = 0
        data.loc[t_df.loc[(t_df['Stance'] == 'CON'), :].index.unique(0), 'stance_eval'] = -1

        data.dropna(axis=0, inplace=True)

    # data = data.sample(10)

    if len(query) > 0:
        with fidx:
            for image_id in data.index:
                data.loc[image_id, 'query_sentiment'] = sentiment_detection.sentiment_nltk(' '.join(query))

                data.loc[image_id, 'query_html_eq'] = query_frequency(query, fidx.get_html_text(image_id))
                data.loc[image_id, 'query_image_eq'] = query_frequency(query, fidx.get_image_text(image_id))
                data.loc[image_id, 'query_html_context'] = context_sentiment(query, fidx.get_html_text(image_id))
                data.loc[image_id, 'query_image_context'] = context_sentiment(query, fidx.get_image_text(image_id))
                data.loc[image_id, 'query_image_align'] = alignment_query(query, fidx.get_image_text(image_id))
    return data


def scale_data(data: pd.DataFrame) -> pd.DataFrame:
    scaled = data.copy()
    scaled = _scale_column(scaled, 'image_percentage_green', 100)
    scaled = _scale_column(scaled, 'image_percentage_red', 100)
    scaled = _scale_column(scaled, 'image_percentage_blue', 100)
    scaled = _scale_column(scaled, 'image_percentage_yellow', 100)
    scaled = _scale_column(scaled, 'image_percentage_bright', 100)
    scaled = _scale_column(scaled, 'image_percentage_dark', 100)
    scaled = _scale_column(scaled, 'image_average_color_r', 360)
    scaled = _scale_column(scaled, 'image_average_color_g', 360)
    scaled = _scale_column(scaled, 'image_average_color_b', 360)
    scaled = _scale_column(scaled, 'image_dominant_color_r', 360)
    scaled = _scale_column(scaled, 'image_dominant_color_g', 360)
    scaled = _scale_column(scaled, 'image_dominant_color_b', 360)

    scaled = scaled.assign(text_len=scaled['text_len'].map(_text_len_scale),
                           image_roi_area=scaled['image_roi_area'].map(_log_normal_density_function),
                           )

    _split_neg(scaled, 'text_sentiment_score')
    _split_neg(scaled, 'query_sentiment')
    _split_neg(scaled, 'html_sentiment_score')
    _split_neg(scaled, 'query_html_context')
    _split_neg(scaled, 'query_image_context')

    return scaled


def _scale_column(data: pd.DataFrame, name: str, factor: float) -> pd.DataFrame:
    if name in data.columns:
        return data.assign(**{name: (data[name]/factor)})
    return data


def _split_neg(data: pd.DataFrame, name: str) -> None:
    if name in data.columns:
        data.loc[:, f'{name}_con'] = data[name] * -1
        data.loc[data[name] > 0, f'{name}_con'] = 0
        data.loc[data[name] < 0, name] = 0


def _log_normal_density_function(x: float) -> float or None:
    if x is None:
        return None
    elif x == 0 or x == 1:
        return 0
    else:
        return ((1 / (math.sqrt(2 * math.pi) * 0.16 * (-x + 1))) * math.exp(
            ((math.log((-x + 1), 10) + 0.49) ** 2) / -0.0512) * 0.12)


def _text_len_scale(x: float) -> float:
    return 1 - (1 / (math.exp(0.01 * x)))
