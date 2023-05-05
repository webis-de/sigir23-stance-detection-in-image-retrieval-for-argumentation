import logging
from typing import List, Tuple

import numpy as np
import pandas as pd


class StanceModel:
    log = logging.getLogger('StanceModel')

    def __init__(self):
        """
        Constructor for model base class.
        """
        pass

    def __str__(self):
        return self.__class__.__name__

    def score(self, query: List[str], doc_id: str) -> float:
        """
        Calculates the stance score for a document (given by index and doc_id) and query (give ans query term list).
        Interpretation:
            score < 0  => doc_id is CON
            score = 0  => doc_id is NONE
            score > 0  => doc_id is PRO

        :param query: preprocessed query in list representation to calculate the relevance for
        :param doc_id: document to calculate the relevance for
        :return: stance score
        """
        return 0.0

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
        if len(argument_relevant) == 0:
            return argument_relevant.copy(), argument_relevant.copy()

        argument_relevant.loc[:, 'stance'] = np.nan
        pro_scores = argument_relevant.copy()
        con_scores = argument_relevant.copy()
        if top_k < 0:
            top_k = len(argument_relevant.index)
        else:
            top_k = min(len(argument_relevant.index), top_k)

        for doc_id in argument_relevant.index:
            score = self.score(query, doc_id)
            argument_relevant.loc[doc_id, 'stance'] = score
            if score > 0:
                pro_scores.loc[doc_id, 'stance'] = score
            elif score < 0:
                con_scores.loc[doc_id, 'stance'] = abs(score)

        return pro_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all'), \
            con_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all')
