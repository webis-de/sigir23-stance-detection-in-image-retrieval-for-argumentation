import logging
from typing import Literal, List

import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from indexing import FeatureIndex, Topic, StanceNetwork
from indexing.neural_net import image_preprocess
from retrieval import RetrievalSystem

from .eval_runs import evaluation, get_save_path
from config import Config

log = logging.getLogger('Crossvalidation')


def crossvalidation_neural_stance_model(version: int):
    fidx = FeatureIndex.load('23826')
    topics_no = [1, 2, 4, 8, 9, 10, 15, 20, 21, 22, 27, 31, 33, 36, 37, 40, 43, 45, 47, 48]
    topics = [Topic.get(t) for t in topics_no]

    img_data = image_preprocess.preprocessed_data(fidx, topics, train=True, with_query=True)
    img_data = image_preprocess.scale_data(img_data)

    for topic in tqdm(topics, desc=f'Crossvalidation StanceNetworkV{version}'):
        data = img_data.loc[img_data['topic'] != topic.number, :]
        StanceNetwork.get(f'cv_missing_{topic.number}', version=version).train(data, test=[])

        evaluation(f'webis#0.5:elastic#0.5:NN-V3-model_1#0.0:NN-V{version}-cv_missing_{topic.number}')

    StanceNetwork.get(f'cv_all_topics', version=version).train(img_data, test=[])
    evaluation(f'webis#0.5:elastic#0.5:NN-V3-model_1#0.0:NN-V{version}-cv_all_topics')


def n_fold_stance_model(versions: List[int], n: int, train_on: Literal['aramis', 'touche-qrels'] = 'aramis'):
    fidx = FeatureIndex.load('23826')
    topics = Topic.load_all()

    if len(topics) % n != 0:
        raise ValueError(f'Topics ({len(topics)}) cant be equally divided into {n} folds.')

    fold_size = int(len(topics) / n)
    folds = [topics[i:i+fold_size] for i in range(0, len(topics), fold_size)]

    cfg = Config.get()
    save_p = cfg.working_dir.joinpath('image_preprocess_df.pkl')
    if save_p.exists():
        img_data = pd.read_pickle(save_p)
    else:
        img_data = image_preprocess.preprocessed_data(fidx, topics, train=True, with_query=True, train_on=train_on)
        img_data = image_preprocess.scale_data(img_data)
        img_data.to_pickle(save_p)

    for version in tqdm(versions, desc='Different Versions:'):
        run_df = pd.DataFrame(columns=['topic', 'stance', 'image_id', 'rank', 'score', 'method'])
        method_tag_sub = f'webis#0.5:elastic#0.5:NN-V3-model_1#0.0:NN-V{version}-'
        save_path = get_save_path(method_tag_sub + f'{n}_folds-{len(topics)}_topics')

        for i, fold in tqdm(enumerate(folds), desc=f'{n} Fold Crossvalidation StanceNetworkV{version}',
                            total=len(folds)):
            topic_nos = [topic.number for topic in fold]
            data = img_data.loc[img_data['topic'].isin(topic_nos), :]
            model_name = f'{n}_folds-{len(topics)}_topics-fold_{i}'
            if version == 7:
                pro = StanceNetwork.get(model_name+'_pro', version=7)
                pro.train(data, test=[])

                con = StanceNetwork.get(model_name+'_con', version=7)
                con.stance = 'CON'
                con.train(data, test=[])
            else:
                StanceNetwork.get(model_name, version=version).train(data=data, test=[])

            rs = RetrievalSystem.parse_method_tag(method_tag=method_tag_sub + model_name, only_eval_images=True)

            fold_df = rs.qrel_scoring(method_tag=method_tag_sub + model_name,
                                      save_path=save_path.joinpath(f"run_fold_{i}.txt"),
                                      topics=topic_nos)
            run_df = pd.concat([run_df, fold_df])
            tf.keras.backend.clear_session()

        save_path = save_path.joinpath('combined_run.txt')
        run_df.to_csv(save_path, sep=' ', header=False, index=False)
        log.info('Crossvalidation with method tag %s saved under %s', method_tag_sub, save_path)
