from typing import List, Tuple

import numpy as np
import pandas as pd

from indexing import FeatureIndex
from indexing import StanceNetwork, preprocess_data, scale_data
from indexing.neural_net import image_preprocess
from .base_model import StanceModel


class NeuralStanceModel(StanceModel):
    network: StanceNetwork
    index: FeatureIndex

    def __init__(self, index: FeatureIndex, model_name: str, version: int = 3, image_net: bool = False):
        """
        Constructor for model base class,
        :param index: index to get relevance data from
        """
        super().__init__()
        self.index = index
        self.image_net = image_net
        self.network = StanceNetwork.load(model_name, version)

    def __str__(self):
        return f'NeuralStanceModel(V{self.network.version}, {self.network.name})'

    def strip_stance(self, df: pd.DataFrame, min_k: int = 10) -> pd.DataFrame:
        stance = df['stance'].sort_values(ascending=False)

        # diff = (stance - stance.shift(1, fill_value=0)).iloc[1:]
        # striped = diff.iloc[min_k:].loc[diff < -0.01]

        striped = stance.iloc[min_k:].loc[stance < 0.5]
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
        :return: Tuple of given DataFrame with a additional column for pro/con stance score.
            Frames are sorted and reduced to top_k rows
        """
        self.log.debug('start stance process for query %s', query)

        if not self.image_net:
            data = preprocess_data(self.index, argument_relevant.index.unique(0).to_list(), query)
            results = self.network.predict(scale_data(data), query=query)
        else:
            data = image_preprocess.preprocess_data(self.index, argument_relevant.index.unique(0).to_list(), query)
            results = self.network.predict(image_preprocess.scale_data(data), query=query)

        argument_relevant.loc[:, 'stance'] = np.nan
        pro_scores = argument_relevant.copy()
        pro_scores = pro_scores.assign(stance=pd.Series(results[0], index=data.index))
        pro_scores = self.strip_stance(pro_scores)
        con_scores = argument_relevant.copy()
        con_scores = con_scores.assign(stance=pd.Series(results[1], index=data.index))
        # con_scores.loc[:, 'stance'] = pd.Series(results[1], index=data.index)
        con_scores = self.strip_stance(con_scores)

        if top_k < 0:
            top_k = len(pro_scores)
        else:
            top_k = min(len(pro_scores), top_k)

        return pro_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all'), \
            con_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all')
