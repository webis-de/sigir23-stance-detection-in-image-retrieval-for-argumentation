from typing import List, Tuple

import numpy as np
import pandas as pd

from indexing import FeatureIndex
from indexing import StanceNetwork
from .base_model import StanceModel


class DummyStanceModel(StanceModel):
    network: StanceNetwork
    index: FeatureIndex

    def __init__(self):
        """
        Constructor for model base class,
        """
        super().__init__()

    def query(self, query: List[str], argument_relevant: pd.DataFrame,
              top_k: int = -1, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Queries a given preprocessed query against the index using a model scoring function

        :param argument_relevant: DataFrame with data for topic and argument score
        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: Tuple of given DataFrame with an additional column for pro/con stance score.
            Frames are sorted and reduced to top_k rows
        """
        self.log.debug('start stance process for query %s', query)

        if top_k < 0:
            top_k = len(argument_relevant)
        else:
            top_k = min(len(argument_relevant), top_k)

        if 'stance' not in argument_relevant.columns:
            argument_relevant.loc[:, 'stance'] = 0.0
            pro_scores = argument_relevant.copy()
            con_scores = argument_relevant.copy()
        else:
            pro_scores = argument_relevant.loc[argument_relevant['stance'] > 0, :]
            con_scores = argument_relevant.loc[argument_relevant['stance'] < 0, :]
            con_scores = con_scores.assign(stance=np.absolute(con_scores['stance']))

        return pro_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all'), \
            con_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all')
