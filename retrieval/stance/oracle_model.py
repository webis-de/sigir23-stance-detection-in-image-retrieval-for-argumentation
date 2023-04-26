import logging
from typing import List, Dict, Set, Tuple, Union

import numpy as np
import pandas as pd

from config import Config
from .base_model import StanceModel
from evaluation.analysis_labeled_data import qrels_to_labeled_data

cfg = Config.get()


class OracleStanceModel(StanceModel):
    log = logging.getLogger('OracleStanceModel')

    labeled_data: pd.DataFrame

    def __init__(self):
        """
        Constructor for oracle stance model.
        """
        super().__init__()
        self.labeled_data = qrels_to_labeled_data()

    def score(self, query: List[str], doc_id: str, topic: str = None) -> Union[float, None]:
        if topic is None:
            return 0
        try:
            doc_df: pd.DataFrame = self.labeled_data.loc[(doc_id, slice(None), topic), :]
            if len(doc_df.index) > 1:
                print(doc_df)
                pass
            for idx, row in doc_df.iterrows():
                if row['Topic_correct']:
                    if row['Stance'] == 'PRO':
                        return 1
                    elif row['Stance'] == 'CON':
                        return -1
                    return 0

        except KeyError:
            self.log.warning('Found image not in labeled data scores %s', doc_id)
        return np.nan

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
        argument_relevant.loc[:, 'stance'] = np.nan
        argument_relevant.loc[:, 'topic'] = 0.0
        argument_relevant.loc[:, 'argument'] = 0.0
        pro_scores = argument_relevant.copy()
        con_scores = argument_relevant.copy()
        if top_k < 0:
            top_k = len(argument_relevant.index)
        else:
            top_k = min(len(argument_relevant.index), top_k)

        topic = None
        if 'topic' in kwargs.keys():
            topic = kwargs.pop('topic')

        for doc_id in argument_relevant.index:
            score = self.score(query, doc_id, topic=topic)
            argument_relevant.loc[doc_id, 'stance'] = score
            if score > 0:
                pro_scores.loc[doc_id, 'stance'] = score
                pro_scores.loc[doc_id, 'argument'] = 1
                con_scores.loc[doc_id, 'stance'] = 0
                con_scores.loc[doc_id, 'argument'] = 0.5
            elif score < 0:
                con_scores.loc[doc_id, 'stance'] = abs(score)
                con_scores.loc[doc_id, 'argument'] = 1
                pro_scores.loc[doc_id, 'stance'] = 0
                pro_scores.loc[doc_id, 'argument'] = 0.5

        # TODO pro nach con falls con < top_k

        return pro_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all'), \
            con_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all')
