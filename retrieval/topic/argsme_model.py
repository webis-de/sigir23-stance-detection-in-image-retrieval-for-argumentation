import logging
from typing import List, Tuple, Dict

import pandas as pd
import requests

from indexing import ElasticSearchIndex, SpacyPreprocessor
from .base_model import TopicModel


def args_response(query: str) -> List[Tuple[str, int]]:
    response_json = requests.get(
        f'https://www.args.me/api/v2/arguments?query={query.replace(" ", "+")}&format=json&pageSize=10'
        f'&fields=arguments.stance,arguments.premises.text,arguments.premises.annotations'
    ).json()

    result = []

    for arg in response_json['arguments']:
        arg_text = arg['premises'][0]['text']
        snippet = ''
        for annotation in arg['premises'][0]['annotations']:
            if annotation['type'] == 'me.args.argument.Snippet':
                snippet += arg_text[annotation['start']:annotation['end']]

        stance = 1 if arg['stance'] == 'PRO' else -1
        result.append((snippet, stance))
    return result


class ArgsMeTopicModel(TopicModel):
    log = logging.getLogger('ArgsTopicModel')

    elastic_index = ElasticSearchIndex
    merge_way: str

    def __init__(self, elastic_index: ElasticSearchIndex, merge: str = 'max'):
        """
        Constructor for model base class,
        merge = MAX, MIN, SUM, AVG
        """
        super().__init__()
        self.elastic_index = elastic_index
        self.merge_way = merge

    def _merge_scores(self, scores: List[Tuple[str, float, int]]) -> List[Tuple[str, float, int]]:
        id_dict: Dict[str, List[Tuple[float, int]]] = {}
        for image_id, score, stance in scores:
            if image_id in id_dict:
                id_dict[image_id].append((score, stance))
            else:
                id_dict[image_id] = [(score, stance)]

        result = []
        for image_id in id_dict.keys():
            if len(id_dict[image_id]) == 1:
                result.append((image_id, *(id_dict[image_id][0])))
                continue

            pros = []
            cons = []
            for score, stance in id_dict[image_id]:
                if stance < 0:
                    cons.append(score)
                elif stance > 0:
                    pros.append(score)

            if len(pros) > len(cons):
                eval_list = pros
                stance = 1
            else:
                eval_list = cons
                stance = -1

            if self.merge_way == 'MAX':
                score = max(eval_list)
            elif self.merge_way == 'MIN':
                score = min(eval_list)
            elif self.merge_way == 'AVG':
                score = sum(eval_list) / len(eval_list)
            elif self.merge_way == 'SUM':
                score = sum(eval_list)
            else:
                score = max(eval_list)

            result.append((image_id, score, stance))

        return result

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
            top_k = 400
        else:
            top_k = min(400, top_k)

        query_text = kwargs.pop('full_query')
        if query_text is None:
            query_text = ' '.join(query)

        response = args_response(query_text)
        prep = SpacyPreprocessor()

        data = [('I5846', 39.5, 1), ('I5636', 34.5, -1)]
        for argument_text, stance in response:
            scores = self.elastic_index.elastic_query(' '.join(prep.preprocess(argument_text)), 40)
            data += [(image_id, score, stance) for image_id, score in scores]

        scores = self._merge_scores(data)

        df = pd.DataFrame(scores, columns=['image_id', 'topic', 'stance'])
        df.set_index('image_id', drop=True, inplace=True)

        return df.nlargest(top_k, 'topic', keep='all')
