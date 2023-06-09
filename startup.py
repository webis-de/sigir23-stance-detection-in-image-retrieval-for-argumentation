import argparse
import datetime
import json
import logging
import pathlib
import sys
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

from config import Config
from frontend import start_server
from indexing import FeatureIndex, Topic, preprocessed_data, DataEntry, ElasticSearchIndex, StanceNetwork, \
    SpacyPreprocessor
from retrieval import RetrievalSystem, ElasticSearchTopicModel, NeuralArgumentModel, FormulaArgumentModel, \
    RandomStanceModel, GoogleStanceModel, BertStanceModel, NeuralStanceModel, FormulaStanceModel, \
    DummyStanceModel, DummyArgumentModel, ArgsMeTopicModel
from utils import setup_logger_handler
from indexing.neural_net import image_preprocess
from evaluation import eval_runs, crossvalidation_neural_stance_model, n_fold_stance_model
from indexing.feature.html_preprocessing import run_html_preprocessing
from evaluation.analysis_labeled_data import labeled_data_to_qrels, create_eda_md_table, exploratory_data_analysis

args: Dict[str, Any] = None


def init_logging():
    """
    Method where the root logger is setup
    """

    root = logging.getLogger()
    setup_logger_handler(root)
    root.setLevel(logging.DEBUG)
    #root.setLevel(logging.INFO)

    root.info('Logging initialised')
    root.debug('Set to debug level')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=pathlib.Path,
                        dest='data_dir', help='Path to input directory.')
    parser.add_argument("-o", "--output-dir", type=pathlib.Path,
                        dest='out_dir', help='Path to output directory.')

    parser.add_argument("-w", "--working-dir", type=pathlib.Path,
                        dest='work_dir', help='Path to working directory. (Location of index/neural net models)')
    parser.add_argument("-cfg", "--config", default=pathlib.Path('config.json'), type=pathlib.Path,
                        dest='config', help='Path to config.json file.')
    parser.add_argument("-f", "--image_format", action='store_true',
                        dest='image_format', help='Specifies format of input data. See README for definition.')

    parser.add_argument('-c', '--count_images', action='store_true',
                        dest='count_ids', help='Performs a count of found images in input.')
    parser.add_argument('-idx', '--indexing', action='store_true', dest='indexing',
                        help='Calculate the index on the given input.')
    parser.add_argument('-esidx', '--elastic-indexing', action='store_true', dest='elastic_indexing',
                        help='Calculate the elastic index on the given input.')
    parser.add_argument('-tidx', '--test-indexing', action='store_true', dest='test_indexing',
                        help='Perform a small indexing run with only 5 images to test the indexing.')
    parser.add_argument('-njobs', '--number-jobs', type=int, dest='n_jobs', default=-1,
                        help='Number of processors to use in parallel indexing process. -1 = all Processors,'
                             ' -2 = all processors but one')
    parser.add_argument('-qrel', '--qrel', action='store_true', dest='qrel',
                        help='Perform a retrieval run over all topics and create run.txt')
    parser.add_argument('-cv', '--crossvalidation', type=int, dest='cv',
                        help='Perform a crossvalidation over the StanceNetwork for the given version')
    parser.add_argument('-mtag', '--method_tag', type=str, dest='method_tag',
                        default='webis#0.5:elastic#0.5:NN-V3-model_1#0.0:random',
                        help='Retrieval method tag for retrieval run.\n'
                             'Format: "webis#{topicWeight}:{TopicModel}#{argumentWeight}:{ArgumentModel}'
                             '#{stanceWeight}:{StanceModel} where:\n"'
                             '  - topicWeight is the weight of the topic score (float in [0,1])\n'
                             '  - TopicModel is "elastic" or "argsme"\n'
                             '  - argumentWeight is the weight of the argument score (float in [0,1])\n'
                             '  - ArgumentModel is "dummy", "formula" or "NN-V{network_version}-{model_name}"\n'
                             '  - stanceWeight is the weight of the stance score (float in [0,1])\n'
                             '  - StanceModel is "dummy", "formula", "bert", "random", "google" or '
                             '"NN-V{network_version}-{model_name}"\n'
                        )

    parser.add_argument('-web', '--website', action='store_true', dest='frontend',
                        help='Start flask web server.')
    parser.add_argument('-p', '--port', type=int, dest='port', default=5000,
                        help='Port for web server.')
    parser.add_argument('-host', '--host', type=str, dest='host', default='0.0.0.0',
                        help='Host address for web server.')

    global args
    args = parser.parse_args()
    args = vars(args)

    if 'config' in args.keys():
        Config._save_path = args['config']

    cfg = Config.get()
    if 'data_dir' in args.keys() and args['data_dir'] is not None:
        cfg.data_dir = args['data_dir']
    if 'out_dir' in args.keys() and args['out_dir'] is not None:
        cfg.output_dir = args['out_dir']
    if 'work_dir' in args.keys() and args['work_dir'] is not None:
        cfg.working_dir = args['work_dir']
    if 'image_format' in args.keys():
        cfg.data_image_format = args['image_format']

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.working_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()


def handle_args():
    if args['count_ids']:
        log.info('Found %s images in data.', len(DataEntry.get_image_ids()))
        sys.exit(0)

    if args['test_indexing']:
        then = datetime.datetime.now()
        log.info('Start feature index creation for %s images', 5)
        Topic.load_all()[0].get_image_ids()
        fidx = FeatureIndex.create_index(name='test', max_images=5, n_jobs=args['n_jobs'])
        fidx.save()
        log.info('Start term index creation for %s images', 5)
        esidx = ElasticSearchIndex('ecir_html_ocr')
        esidx.create_index_with_features(fidx)
        log.info('Precalculate data for retrieval process')
        preprocessed_data(fidx, Topic.load_all())
        dur = datetime.datetime.now() - then
        log.info('Time for index creation %s', dur)
        sys.exit(0)

    if args['indexing']:
        max_id = len(DataEntry.get_image_ids())
        index_creation(max_id, n_jobs=args['n_jobs'])
        sys.exit(0)

    if args['elastic_indexing']:
        log.info('Start elastic index creation for 23826 images')
        esidx = ElasticSearchIndex(f'ecir_html_ocr_v2')
        esidx.create_index_with_features(FeatureIndex.load('23826'))
        sys.exit(0)

    if args['qrel']:
        log.info('Start qrel scoring with method tag %s', args['method_tag'])
        RetrievalSystem.parse_method_tag(args['method_tag']).qrel_scoring(args['method_tag'])
        sys.exit(0)

    if args['cv']:
        log.info('Start StanceNetwork crossvalidation for version %s', args['cv'])
        crossvalidation_neural_stance_model(args['cv'])
        sys.exit(0)

    if args['frontend']:
        log.info('Start flask frontend with method tag %s', args['method_tag'])
        rs = RetrievalSystem.parse_method_tag(args['method_tag'])
        start_server([rs.topic_model], [rs.argument_model], [rs.stance_model], host=args['host'], port=args['port'])
        sys.exit(0)

    main()


def start_flask():
    topic_models = []
    arg_models = []
    stance_models = []
    fidx = FeatureIndex.load('23826')
    esidx = ElasticSearchIndex('ecir_html_ocr_v2')

    topic_models.append(ElasticSearchTopicModel(esidx))
    topic_models.append(ArgsMeTopicModel(esidx))

    arg_models.append(FormulaArgumentModel(fidx))
    arg_models.append(DummyArgumentModel())
    arg_models.append(NeuralArgumentModel(fidx, 'model_1'))

    stance_models.append(RandomStanceModel())
    stance_models.append(DummyStanceModel())
    stance_models.append(GoogleStanceModel())
    stance_models.append(FormulaStanceModel(fidx))
    stance_models.append(BertStanceModel())
    stance_models.append(NeuralStanceModel(fidx, 'model_1', version=3))
    stance_models.append(NeuralStanceModel(fidx, 'model_all_topic', version=5, image_net=True))
    stance_models.append(NeuralStanceModel(fidx, 'model_all_topic', version=6, image_net=True))

    start_server(topic_models, arg_models, stance_models)


def index_creation(max_images: int = -1, n_jobs: int = -2) -> None:
    then = datetime.datetime.now()
    log.info('Start feature index creation for %s images', max_images)
    Topic.load_all()[0].get_image_ids()
    fidx = FeatureIndex.create_index(name='complete', max_images=max_images, n_jobs=n_jobs)
    fidx.save()
    log.info('Start term index creation for %s images', max_images)
    esidx = ElasticSearchIndex(f'ecir_html_ocr_v2')
    esidx.create_index_with_features(fidx)
    log.info('Precalculate data for retrieval process')
    preprocessed_data(fidx, Topic.load_all())
    dur = datetime.datetime.now() - then
    log.info('Time for index creation %s', dur)


def test_stance_network():
    fidx = FeatureIndex.load('23826')
    topics_no = [1, 2, 4, 8, 9, 10, 15, 20, 21, 22, 27, 31, 33, 36, 37, 40, 43, 45, 47, 48]
    # topics_no = [20, 43, 47, 40, 45, 33, 2, 15, 10, 36]
    # topics_no = [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31,
    #              32, 33, 35, 37, 38, 40, 41, 43, 44, 45, 46, 47, 49, 50]

    topics = [Topic.get(t) for t in topics_no]

    topics_train = [1, 2, 4, 8, 9, 10, 15, 20, 21, 22, 27, 31, 33, 36, 37, 40, 43, 45, 47, 48]
    topics_eval = [topic.number for topic in Topic.load_all() if topic.number not in topics_train]

    # img_data = image_preprocess.preprocessed_data(fidx, topics, train=True, with_query=False)
    # img_data = image_preprocess.scale_data(img_data)

    # pro = StanceNetwork.get('all_topics_pro', version=7)
    # pro.train(img_data, test=[])
    #
    # con = StanceNetwork.get('all_topics_con', version=7)
    # con.stance = 'CON'
    # con.train(img_data, test=[])

    eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.5:NN-V7-all_topics', topics=topics_eval)

    def calc_network(name, version):
        StanceNetwork.get(name, version=version).train(img_data, test=[])
        eval_runs.evaluation(f'webis#0.5:elastic#0.5:NN-V3-model_1#0.0:NN-V{version}-{name}', topics=topics_eval)

    # try:
    #     calc_network('model_all_topic', 6)
    # except Exception as e:
    #     log.error(e, exc_info=True)
    #
    # try:
    #     calc_network('model_all_topic', 8)
    # except Exception as e:
    #     log.error(e, exc_info=True)
    #
    # try:
    #     calc_network('model_all_topic', 9)
    # except Exception as e:
    #     log.error(e, exc_info=True)


def main():
    """
    normal program run
    :return:
    """

    log.info('do main stuff')

    n_fold_stance_model([7], 5, 'touche-qrels')

    # analysis, eda = exploratory_data_analysis()
    # create_eda_md_table(analysis, eda)

    # topics_train = [1, 2, 4, 8, 9, 10, 15, 20, 21, 22, 27, 31, 33, 36, 37, 40, 43, 45, 47, 48]
    topics_train = []
    topics_eval = [topic.number for topic in Topic.load_all() if topic.number not in topics_train]

    # eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:clip', topics=topics_eval)

    # eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.5:NN-V6-model_all_topic', topics=topics_eval)
    # eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:oracle', topics=topics_eval)

    try:
        eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:afinn', topics=topics_eval)
        eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:bert', topics=topics_eval)
        eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:clip', topics=topics_eval)
        eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:dummy', topics=topics_eval)
        eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:formula', topics=topics_eval)
        eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:google', topics=topics_eval)
        eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:oracle', topics=topics_eval)
        eval_runs.evaluation('webis#0.5:elastic#0.5:NN-V3-model_1#0.0:random', topics=topics_eval)
    except Exception as e:
        log.error(e, exc_info=True)

    # test_stance_network()

    pass


if __name__ == '__main__':
    parse_args()
    init_logging()
    log = logging.getLogger('startup')
    try:
        handle_args()
    except Exception as e:
        log.error(e, exc_info=True)
        raise
