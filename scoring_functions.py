import json

import pandas as pd
from tqdm import tqdm

from config import Config
from indexing import FeatureIndex, Topic, SpacyPreprocessor
from indexing.feature.html_preprocessing import run_html_preprocessing
from retrieval import NeuralArgumentModel, FormulaArgumentModel, \
    RandomStanceModel, GoogleStanceModel, BertStanceModel, NeuralStanceModel, FormulaStanceModel


def stuff():
    # eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:random', topics=topics_eval)
    # eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:bert', topics=topics_eval)
    # eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:formula', topics=topics_eval)
    # eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:google', topics=topics_eval)
    # eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:NN-V3-model_1', topics=topics_eval)

    # eval_runs.evaluation('webis#0.5:argsme#0.5:dummy#0.0:dummy', topics=topics_eval)

    fidx = FeatureIndex.load('23826')
    cfg = Config.get()
    with fidx:
        with cfg.output_dir.joinpath('image_full_text_ocr.jsonl').open(mode='w') as f:
            for image_id in tqdm(fidx.get_image_ids()):
                doc_text = {'image_id': image_id, 'html_text': run_html_preprocessing(image_id),
                            'ocr_text': fidx.get_image_text(image_id)}
                f.write(json.dumps(doc_text) + '\n')
            f.flush()


def create_scoring():
    stance_models = []
    fidx = FeatureIndex.load('23826')
    stance_models.append(RandomStanceModel())
    stance_models.append(GoogleStanceModel())
    stance_models.append(FormulaStanceModel(fidx))
    stance_models.append(BertStanceModel())
    stance_models.append(NeuralStanceModel(fidx, 'model_1', version=3))
    stance_models.append(NeuralStanceModel(fidx, 'model_all_topic', version=5, image_net=True))
    stance_models.append(NeuralStanceModel(fidx, 'model_all_topic', version=6, image_net=True))

    arg_models = [FormulaArgumentModel(fidx), NeuralArgumentModel(fidx, 'model_1')]

    cfg = Config.get()
    ground_truth = pd.read_csv(cfg.data_dir.joinpath("touche-task3-001-050-relevance.qrels"), sep=" ",
                               names=["topic", "characteristic", "web_id", "value"])
    df: pd.DataFrame = ground_truth.loc[:, ['topic', 'web_id']]
    for topic in tqdm(df['topic'].unique().tolist()):
        image_ids = df.loc[df['topic'] == topic, 'web_id'].unique().tolist()
        image_ids = [img_id for img_id in image_ids if img_id in fidx.get_image_ids()]
        topic_df = pd.DataFrame(image_ids, columns=['image_id'])
        topic_df.loc[:, 'stance'] = 0.0
        topic_df.set_index('image_id', drop=True, inplace=True)
        query = SpacyPreprocessor().preprocess(Topic.get(topic).title)
        for stance_model in stance_models:
            stance_model.query(query, topic_df)
            topic_df = topic_df.assign(**{str(stance_model): topic_df['stance']})
        for arg_model in arg_models:
            arg_model.query(query, topic_df)
            topic_df = topic_df.assign(**{arg_model.__class__.__name__: topic_df['argument']})

        path = cfg.output_dir.joinpath(f'stance_scores')
        path.mkdir(exist_ok=True, parents=True)
        with path.joinpath(f'topic_{topic}.jsonl').open(mode='w') as f:
            for image_id, row in topic_df.iterrows():
                row = row.loc[row.index != 'stance']
                row = row.loc[row.index != 'argument']
                doc_text = {'image_id': image_id, 'topic_id': topic, **row.to_dict()}
                f.write(json.dumps(doc_text) + '\n')


def combine_stance_scores():
    cfg = Config.get()
    ground_truth = pd.read_csv(cfg.data_dir.joinpath("touche-task3-001-050-relevance.qrels"), sep=" ",
                               names=["topic", "characteristic", "web_id", "value"])
    df: pd.DataFrame = ground_truth.loc[:, ['topic', 'web_id']]
    path = cfg.output_dir.joinpath(f'stance_scores')
    with path.joinpath(f'topic_all.jsonl').open(mode='w') as f:
        for topic in tqdm(df['topic'].unique().tolist()):
            with path.joinpath(f'topic_{topic}.jsonl').open(mode='r') as topic_file:
                f.writelines(topic_file.readlines())
