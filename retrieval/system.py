import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from config import Config
from indexing import FeatureIndex, Topic, ElasticSearchIndex, SpacyPreprocessor
from indexing import Preprocessor
from .argument import ArgumentModel, FormulaArgumentModel, NeuralArgumentModel, DummyArgumentModel
from .stance import StanceModel, RandomStanceModel, FormulaStanceModel, NeuralStanceModel, BertStanceModel, \
    DummyStanceModel, GoogleStanceModel, ParallelNeuralStanceModel, AFinnStanceModel, OracleStanceModel, ClipStanceModel
from .topic import TopicModel, ElasticSearchTopicModel, ArgsMeTopicModel

log = logging.getLogger('retrievalSystem')
cfg = Config.get()


class RetrievalSystem:

    def __init__(self, prep: Preprocessor, topic_model: TopicModel, argument_model: ArgumentModel,
                 stance_model: StanceModel, topic_weight: float = 0.10, arg_weight: float = 0.9,
                 stance_weight: float = 0.0, only_eval_images: bool = False):
        """
        RetrievalSystem Constructor. Combines a preprocessor, a TopicModel, an ArgumentModel
        and a StanceModel to a RetrievalSystem. Applies given weight to scores of the different models.

        :param prep: Preprocessor instance, for query preprocessing
        :param topic_model: TopicModel to calculate a topic score
        :param argument_model: ArgumentModel to calculate an argument score
        :param stance_model: StanceModel to calculate a stance score
        :param topic_weight: weight of the topic score in final score
        :param arg_weight: weight of the argument score in final score
        :param stance_weight: weight of the stance score in final score
        :param only_eval_images: Should the retrieval only return images where an evaluation exist
        """
        self.prep = prep
        self.topic_model = topic_model
        self.argument_model = argument_model
        self.stance_model = stance_model
        self.topic_weight = topic_weight
        self.arg_weight = arg_weight
        self.stance_weight = stance_weight
        self.only_eval_image = only_eval_images

    def query(self, text: str, top_k: int = -1, **kwargs) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Queries a given text against the index
        :param text: query text
        :param top_k: number of top results to return
        :return: (list of pro (doc_id, score), list of con (doc_id, score))
         tuples descending by score for all documents in the vector space
        """
        log.debug('start retrieval for query "%s"', text)
        query = self.prep.preprocess(text)

        topic = None
        if 'topic' in kwargs.keys():
            topic = kwargs.pop('topic')

        topic_scores = self.topic_model.query(query, full_query=text, topic=topic,
                                              only_judged=self.only_eval_image)
        argument_scores = self.argument_model.query(query, topic_scores, **kwargs)
        pro_scores, con_scores = self.stance_model.query(query, argument_scores, topic=topic, full_query=text, **kwargs)
        # pro_scores, con_scores = self.stance_model.query(query, argument_scores, top_k=int(6*top_k), **kwargs)
        # 3 CON best, 6 PRO & all best

        if top_k < 0:
            top_k = len(topic_scores)

        pro = pd.DataFrame(minmax_scale(pro_scores), columns=pro_scores.columns, index=pro_scores.index)
        con = pd.DataFrame(minmax_scale(con_scores), columns=con_scores.columns, index=con_scores.index)

        ps = self.topic_weight * pro['topic'] + self.arg_weight * pro['argument'] + self.stance_weight * pro['stance']
        cs = self.topic_weight * con['topic'] + self.arg_weight * con['argument'] + self.stance_weight * con['stance']

        ns = pd.Series([0.0] * top_k, index=['None'] * top_k)
        ps = pd.concat([ps, ns])
        cs = pd.concat([cs, ns])

        return [e for e in ps.nlargest(top_k).items()], [e for e in cs.nlargest(top_k).items()]

    def qrel_scoring(self, method_tag: str, save: bool = True, save_path: Path = None,
                     topics: List[int] = None) -> pd.DataFrame:
        """
        Perform a qrel scoring and create a run.txt.
        Returns the created run.txt in a Dataframe.
        The default save path is out/run.txt, an existing file is overridden!

        :param topics: List of topics to score
        :param method_tag: method tag to write in the run.txt
        :param save: should the run.txt be saved?
        :param save_path: Path/to/run.txt or any other file name.
        :return: the created Dataframe for the run.txt
        """
        data = []
        log.info('start qrel scoring')
        if topics is None:
            topics = Topic.load_all()
        else:
            topics = [Topic.get(topic_id) for topic_id in topics]

        for topic in tqdm(topics, desc=f'qrel Scoring for {method_tag}'):
            result_p, result_c = self.query(topic.title, top_k=10, topic=topic.number)
            for i, r in enumerate(result_p):
                data.append([topic.number, 'PRO', r[0], i + 1, round(r[1], 6), method_tag])
            for i, r in enumerate(result_c):
                data.append([topic.number, 'CON', r[0], i + 1, round(r[1], 6), method_tag])
        df = pd.DataFrame(data, columns=['topic', 'stance', 'image_id', 'rank', 'score', 'method'])
        if save:
            if save_path is None:
                save_path = Config.get().output_dir.joinpath('run.txt')
            df.to_csv(save_path, sep=' ', header=False, index=False)
            log.info('scoring with method tag %s saved under %s', method_tag, save_path)
        return df

    @classmethod
    def parse_method_tag(cls, method_tag: str, only_eval_images: bool = False,
                         index_name: str = 'complete', elastic_name: str = 'ecir_html_ocr_v2') -> 'RetrievalSystem':
        """
        Create RetrievalSystem for given method tag.
        Method tag should have format
            webis#{topicWeight}:{TopicModel}#{argumentWeight}:{ArgumentModel}#{stanceWeight}:{StanceModel}
         where:
         - topicWeight is the weight of the topic score (float in [0,1])
         - TopicModel is 'elastic' or 'argsme'
         - argumentWeight is the weight of the argument score (float in [0,1])
         - ArgumentModel is 'dummy', 'formula' or 'NN-V{network_version}-{model_name}'
         - stanceWeight is the weight of the stance score (float in [0,1])
         - StanceModel is 'dummy', 'formula', 'bert', 'random', 'google' or 'NN-V{network_version}-{model_name}'
        :param only_eval_images: Only retrieve on evaluated images
        :param elastic_name: Name of the elastic search index
        :param index_name: Name of the feature index
        :param method_tag: string to parse
        :return: RetrievalSystem for parsed method tag
        :raise ValueError: if method tag is faulty
        """

        def get_weight(weight_str: str, name: str) -> float:
            try:
                weight = float(weight_str.strip()[1:])
                if not (0 <= weight <= 1):
                    raise ValueError
                return weight
            except ValueError:
                raise ValueError(f'{name} weight {weight_str} is not a number in [0,1]')

        split = method_tag.split('#')
        if len(split) == 4 and split[0] == 'webis':
            # TopicModel
            topic_split = split[1].split(':')
            if len(topic_split) != 2:
                raise ValueError('Topic part of method tag is invalid. Correct format: '
                                 '"webis#{topicWeight}:{TopicModel}#{argumentWeight}:{ArgumentModel}'
                                 '#{stanceWeight}:{StanceModel}"')
            topic_weight = get_weight(topic_split[0], 'Topic')
            esidx = ElasticSearchIndex(elastic_name)
            if topic_split[1] == 'elastic':
                topic_model = ElasticSearchTopicModel(esidx)
            elif topic_split[1] == 'argsme':
                topic_model = ArgsMeTopicModel(esidx)
            else:
                raise ValueError('TopicModel {} not found'.format(topic_split[1]))

            # ArgumentModel
            arg_split = split[2].split(':')
            if len(arg_split) != 2:
                raise ValueError('Argument part of method tag is invalid. Correct format: '
                                 '"webis#{topicWeight}:{TopicModel}#{argumentWeight}:{ArgumentModel}'
                                 '#{stanceWeight}:{StanceModel}"')

            arg_weight = get_weight(arg_split[0], 'Argument')
            fidx = FeatureIndex.load(index_name)

            if arg_split[1] == 'formula':
                arg_model = FormulaArgumentModel(fidx)
            elif arg_split[1] == 'dummy':
                arg_model = DummyArgumentModel()
            elif arg_split[1][:3] == 'NN-':
                if arg_split[1][3] == 'V':
                    try:
                        second_split = arg_split[1].index('-', 4)
                        version = int(arg_split[1][4:second_split])
                        model_name = arg_split[1][second_split+1:]
                        arg_model = NeuralArgumentModel(fidx, model_name=model_name, version=version)
                    except ValueError:
                        raise ValueError('NeuralArgumentModel version is invalid. Correct format: '
                                         '"webis#{topicWeight}:{TopicModel}#{argumentWeight}:{ArgumentModel}'
                                         '#{stanceWeight}:{StanceModel}"')
                else:
                    raise ValueError('NeuralArgumentModel version is missing. Correct format: '
                                     '"webis#{topicWeight}:{TopicModel}#{argumentWeight}:{ArgumentModel}'
                                     '#{stanceWeight}:{StanceModel}"')
            else:
                raise ValueError('ArgumentModel {} not found'.format(arg_split[1]))

            # StanceModel
            stance_split = split[3].split(':')
            if len(stance_split) != 2:
                raise ValueError('Stance part of method tag is invalid. Correct format: '
                                 '"webis#{topicWeight}:{TopicModel}#{argumentWeight}:{ArgumentModel}'
                                 '#{stanceWeight}:{StanceModel}"')

            stance_weight = get_weight(stance_split[0], 'Stance')
            if stance_split[1] == 'formula':
                stance_model = FormulaStanceModel(fidx)
            elif stance_split[1] == 'random':
                stance_model = RandomStanceModel()
            elif stance_split[1] == 'bert':
                stance_model = BertStanceModel()
            elif stance_split[1] == 'afinn':
                stance_model = AFinnStanceModel()
            elif stance_split[1] == 'oracle':
                stance_model = OracleStanceModel()
            elif stance_split[1] == 'dummy':
                stance_model = DummyStanceModel()
            elif stance_split[1] == 'clip':
                stance_model = ClipStanceModel()
            elif stance_split[1] == 'google':
                stance_model = GoogleStanceModel()
            elif stance_split[1][:3] == 'NN-':
                if stance_split[1][3] == 'V':
                    try:
                        second_split = stance_split[1].index('-', 4)
                        version = int(stance_split[1][4:second_split])
                        model_name = stance_split[1][second_split+1:]
                        if version == 7:
                            stance_model = ParallelNeuralStanceModel(fidx, pro_model_name=model_name+'_pro',
                                                                     con_model_name=model_name+'_con', version=version)
                        else:
                            stance_model = NeuralStanceModel(fidx, model_name=model_name, version=version)
                    except ValueError:
                        raise ValueError('NeuralStanceModel version is invalid. Correct format: '
                                         '"webis#{topicWeight}:{TopicModel}#{argumentWeight}:{ArgumentModel}'
                                         '#{stanceWeight}:{StanceModel}"')
                else:
                    raise ValueError('NeuralStanceModel version is missing. Correct format: '
                                     '"webis#{topicWeight}:{TopicModel}#{argumentWeight}:{ArgumentModel}'
                                     '#{stanceWeight}:{StanceModel}"')
            else:
                raise ValueError('StanceModel {} not found'.format(stance_split[1]))

            return cls(SpacyPreprocessor(), topic_model=topic_model, argument_model=arg_model,
                       stance_model=stance_model, topic_weight=topic_weight, arg_weight=arg_weight,
                       stance_weight=stance_weight, only_eval_images=only_eval_images)
        raise ValueError('Method tag "%s" is not correctly formatted. Correct format: '
                         '"webis#{topicWeight}:{TopicModel}#{argumentWeight}:{ArgumentModel}'
                         '#{stanceWeight}:{StanceModel}"')
