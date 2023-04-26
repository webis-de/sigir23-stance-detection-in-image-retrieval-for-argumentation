from typing import List

import pandas as pd

from indexing import ArgumentNetwork, preprocess_data, scale_data
from indexing import FeatureIndex
from .base_model import ArgumentModel


class NeuralArgumentModel(ArgumentModel):
    network: ArgumentNetwork

    def __init__(self, index: FeatureIndex, model_name: str, version: int = 3):
        """
        Constructor for model base class,
        :param index: index to get relevance data from
        """
        super().__init__(index)
        self.network = ArgumentNetwork.load(model_name, version)

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
            top_k = len(self.index)
        else:
            top_k = min(len(self.index), top_k)

        # topic = None
        # if 'topic' in kwargs.keys():
        #     topic = kwargs.pop('topic')
        # data = preprocess_data(self.index, topic_relevant.index.unique(0).to_list(), query, topic=topic)
        data = self.index.dataframe.loc[topic_relevant.index.unique(0).to_list(), :].copy()
        results = self.network.predict(scale_data(data))

        results_df = pd.Series(results, index=data.index)

        for doc_id in topic_relevant.index:
            topic_relevant.loc[doc_id, 'argument'] = results_df.loc[doc_id]

        return topic_relevant.nlargest(top_k, 'argument', keep='all')
