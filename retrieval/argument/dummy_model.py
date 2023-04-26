import logging
from typing import List, Tuple

import pandas as pd

from .base_model import ArgumentModel


class DummyArgumentModel(ArgumentModel):
    log = logging.getLogger('DummyArgumentModel')

    def __init__(self):
        """
        Constructor for model base class
        """
        super().__init__(None)

    def score(self, query: List[str], doc_id: str) -> Tuple[float, float, float, float, float]:
        """
        Calculates the argument score for a document (given by index and doc_id) and query (give ans query term list)
        :param query: preprocessed query in list representation to calculate the relevance for
        :param doc_id: document to calculate the relevance for
        :return: argument score
        """
        pass

    def query(self, query: List[str], topic_relevant: pd.DataFrame,
              top_k: int = -1, **kwargs) -> pd.DataFrame:
        """
        Queries a given preprocessed query against the index using a model scoring function

        :param topic_relevant: DataFrame with data for topic score
        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: given DataFrame with a additional column for argument score.
            Frame is sorted and reduced to top_k rows
        """
        self.log.debug('start argument process for query %s', query)

        if top_k < 0:
            top_k = len(topic_relevant)
        else:
            top_k = min(len(topic_relevant), top_k)

        for doc_id in topic_relevant.index:
            topic_relevant.loc[doc_id, 'argument'] = 0.0

        return topic_relevant.nlargest(top_k, 'argument', keep='all')
