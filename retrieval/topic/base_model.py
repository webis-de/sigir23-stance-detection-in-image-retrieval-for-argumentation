import abc
import logging
from typing import List

import pandas as pd


class TopicModel(abc.ABC):
    log = logging.getLogger('TopicModel')

    def __init__(self):
        """
        Constructor for model base class,
        """
        pass

    def _score(self, query: List[str], doc_id: str) -> float:
        """
        Calculates the relevance score for a document (given by index and doc_id) and query (give ans query term list)
        :param query: preprocessed query in list representation to calculate the relevance for
        :param doc_id: document to calculate the relevance for
        :return: relevance score
        """
        return 1.0

    @abc.abstractmethod
    def query(self, query: List[str], top_k: int = -1, **kwargs) -> pd.DataFrame:
        """
        Queries a given query against the index using a model scoring function

        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: DataFrame with a column for topic score.
            Frame is sorted and reduced to top_k rows
        """
        pass
