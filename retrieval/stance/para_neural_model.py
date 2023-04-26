from typing import List, Tuple

import numpy as np
import pandas as pd

from indexing import FeatureIndex
from indexing import StanceNetwork, preprocess_data, scale_data
from indexing.neural_net import image_preprocess
from .base_model import StanceModel


class ParallelNeuralStanceModel(StanceModel):
    network_pro: StanceNetwork
    network_con: StanceNetwork
    index: FeatureIndex

    def __init__(self, index: FeatureIndex, pro_model_name: str, con_model_name: str, version: int = 3):
        """
        Constructor for model base class,
        :param index: index to get relevance data from
        """
        super().__init__()
        self.index = index
        self.network_pro = StanceNetwork.load(pro_model_name, version)
        self.network_con = StanceNetwork.load(con_model_name, version)

    def __str__(self):
        return f'ParallelNeuralStanceModel(V{self.network_pro.version}, {self.network_pro.name}, {self.network_con.name})'

    def strip_stance(self, df: pd.DataFrame, min_k: int = 10) -> pd.DataFrame:
        stance = df['stance'].sort_values(ascending=False)

        # diff = (stance - stance.shift(1, fill_value=0)).iloc[1:]
        # striped = diff.iloc[min_k:].loc[diff < -0.01]
        max_score = stance.iloc[0]

        striped = stance.iloc[min_k:].loc[stance < max_score/2]
        if len(striped.index) > 0:
            image_id_strip = striped.index[0]  # diff shift
            result_ids = stance.loc[:image_id_strip].iloc[:-1]
            result = df.loc[result_ids.index, :]
        else:
            result = df

        self.log.debug(f'Striped from {len(df.index)} to {len(result.index)}')

        return result

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

        data = image_preprocess.preprocess_data(self.index, argument_relevant.index.unique(0).to_list(), query)
        results_pro = self.network_pro.predict(image_preprocess.scale_data(data), query=query)
        results_con = self.network_con.predict(image_preprocess.scale_data(data), query=query)

        pro_scores = argument_relevant.copy()
        pro_scores.loc[:, 'stance'] = pd.Series(results_pro.flatten(), index=data.index)
        pro_scores = self.strip_stance(pro_scores)
        con_scores = argument_relevant.copy()
        con_scores.loc[:, 'stance'] = pd.Series(results_con.flatten(), index=data.index)
        con_scores = self.strip_stance(con_scores)

        if top_k < 0:
            top_k = len(pro_scores)
        else:
            top_k = min(len(pro_scores), top_k)

        return pro_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all'), \
            con_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all')
