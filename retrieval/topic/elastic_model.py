import logging
from typing import List, Dict, Set

import pandas as pd

from config import Config
from indexing import ElasticSearchIndex
from .base_model import TopicModel

cfg = Config.get()


def create_eval_images() -> Dict[int, Set[str]]:
    truth = pd.read_csv(cfg.data_dir.joinpath("touche-task3-001-050-relevance.qrels"), sep=" ",
                        names=["topic", "characteristic", "web_id", "value"])
    truth = truth.loc[:, ['topic', 'web_id']].drop_duplicates()
    result = {}
    for index, row in truth.iterrows():
        result.setdefault(row['topic'], set()).add(row['web_id'])
    return result


eval_image_ids = create_eval_images()


class ElasticSearchTopicModel(TopicModel):
    log = logging.getLogger('ElasticTopicModel')

    elastic_index = ElasticSearchIndex

    def __init__(self, elastic_index: ElasticSearchIndex):
        """
        Constructor for model base class,
        """
        super().__init__()
        self.elastic_index = elastic_index

    def query(self, query: List[str], top_k: int = -1, **kwargs) -> pd.DataFrame:
        """
        Queries a given query against the index using a model scoring function

        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: DataFrame with a column for topic score.
            Frame is sorted and reduced to top_k rows
        """
        self.log.debug('start topic process for query %s', query)
        if top_k < 0:
            top_k = len(self.elastic_index)
        else:
            top_k = min(len(self.elastic_index), top_k)

        # TODO how many should be precalculated?
        only_judged = kwargs.pop('only_judged')
        topic = kwargs.pop('topic')
        image_ids = None
        if only_judged is not None and only_judged and topic is not None and topic in eval_image_ids.keys():
            image_ids = eval_image_ids[topic]
        scores = self.elastic_index.elastic_query(' '.join(query), 400, image_ids=list(image_ids))
        df = pd.DataFrame(scores, columns=['image_id', 'topic'])
        df.set_index('image_id', drop=True, inplace=True)

        return df.nlargest(top_k, 'topic', keep='all')
