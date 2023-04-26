import logging
import math
from typing import List, Tuple

import numpy as np
import pandas as pd

from indexing import FeatureIndex, ImageType
from indexing.feature import sentiment_detection
from .base_model import StanceModel


class FormulaStanceModel(StanceModel):
    log = logging.getLogger('FormulaStanceModel')

    index: FeatureIndex

    def __init__(self, index: FeatureIndex, weights: List[float] = None):
        """
        Constructor for model base class,
        :param index: index to get relevance data from
        :param weights: weights for query
        """
        super().__init__()
        self.index = index
        self.weights = weights

    def score(self, query: List[str], doc_id: str) -> Tuple[float, float, float]:
        """
        Calculates the stance score for a document (given by index and doc_id) and query (give ans query term list)
        :param query: preprocessed query in list representation to calculate the relevance for
        :param doc_id: document to calculate the relevance for
        :return: stance score
        """

        image_type = self.index.get_image_type(doc_id)

        percentage_green = self.index.get_image_percentage_green(doc_id)
        percentage_red = self.index.get_image_percentage_red(doc_id)
        percentage_bright = self.index.get_image_percentage_bright(doc_id)
        percentage_dark = self.index.get_image_percentage_dark(doc_id)
        html_sentiment_score = self.index.get_html_sentiment_score(doc_id)

        # between 0 and 3
        color_mood = 0
        if image_type == ImageType.CLIPART:
            color_mood = (percentage_green - percentage_red) * 3
        elif image_type == ImageType.PHOTO:
            hue_factor = 0.2
            # between -1 and 1
            color_mood = ((percentage_green / 100 - percentage_red / 100) * (1 - hue_factor)) + \
                         ((percentage_bright / 100 - percentage_dark / 100) * hue_factor)
            # between -3 and 3
            color_mood = color_mood * 3

        image_text_sentiment_score = self.index.get_text_sentiment_score(doc_id)
        image_text_len = self.index.get_text_len(doc_id)
        # between 1 and 3 (above 80 ~3)
        len_words_value = 3 + (((-1) / (math.exp(0.04 * image_text_len))) * 2)
        text_sentiment_factor = len_words_value * image_text_sentiment_score

        # shift between -3 and 3
        html_sentiment_score = html_sentiment_score * 3

        query_sentiment_score = sentiment_detection.sentiment_nltk(' '.join(query))
        query_negation = (query_sentiment_score < 0) if abs(query_sentiment_score) > 0.2 else False

        if query_negation:
            color_mood = color_mood * (-1)
            text_sentiment_factor = text_sentiment_factor * (-1)
            html_sentiment_score = html_sentiment_score * (-1)

        return color_mood, text_sentiment_factor, html_sentiment_score

    def query(self, query: List[str], argument_relevant: pd.DataFrame,
              **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Queries a given preprocessed query against the index using a model scoring function

        :param argument_relevant: DataFrame with data for topic and argument score
        :param query: preprocessed query in list representation to calculate the relevance for
        :return: Tuple of given DataFrame with a additional column for pro/con stance score.
            Frames are sorted and reduced to top_k rows
        """
        self.log.debug('start stance process for query %s', query)
        pro_scores = argument_relevant.copy()
        con_scores = argument_relevant.copy()

        df = pd.DataFrame(index=argument_relevant.index, columns=['color_mood', 'image_text_sentiment_score',
                                                                  'html_sentiment_score'])

        for doc_id in argument_relevant.index:
            df.loc[doc_id, :] = self.score(query, doc_id)

        df_norm = df / df.abs().max()

        if self.weights is None:
            np_weights = np.array([1, 1, 1])
        else:
            np_weights = np.array(self.weights)

        np_weights = np_weights / np_weights.sum()

        for doc_id in argument_relevant.index:
            score = (df_norm.loc[doc_id, :].to_numpy() * np_weights).mean()
            if score > 0:
                argument_relevant.loc[doc_id, 'stance'] = 1
                pro_scores.loc[doc_id, 'stance'] = 1
            elif score < 0:
                con_scores.loc[doc_id, 'stance'] = -1
                argument_relevant.loc[doc_id, 'stance'] = -1
            else:
                argument_relevant.loc[doc_id, 'stance'] = 0

        return pro_scores.dropna(axis=0), con_scores.dropna(axis=0)
