import datetime
import logging
import os
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from flask import Flask, render_template, request, send_file, abort, make_response

from evaluation import save_eval, Stance, Argumentative, get_image_to_eval, has_eval
from indexing import DataEntry, Topic, SpacyPreprocessor
from retrieval import RetrievalSystem, TopicModel, ArgumentModel, StanceModel, DummyStanceModel, NeuralStanceModel
from config import Config

log = logging.getLogger('frontend.flask')
app = Flask(__name__, static_url_path='', static_folder='static')

topic_models: List[TopicModel]
arg_models: List[ArgumentModel]
stance_models: List[StanceModel]
valid_system = False
image_ids: List[str] = []
cfg = Config.get()

ground_truth = pd.read_csv(cfg.data_dir.joinpath("touche-task3-001-050-relevance.qrels"), sep=" ",
                           names=["topic", "characteristic", "image_id", "value"])


def _get_image_eval(image_id: str, topic_id: int) -> int or None:
    """
    -1 -> CON
    0 -> not on Topic
    1 -> PRO
    None -> no eval
    :param image_id:
    :param topic_id:
    :return:
    """
    topic_df: pd.DataFrame = ground_truth.loc[ground_truth['topic'] == topic_id, :]
    if image_id not in topic_df['image_id'].unique():
        return None

    img_df: pd.DataFrame = topic_df.loc[topic_df['image_id'] == image_id, :]
    if (img_df.loc[img_df['characteristic'] == 'ONTOPIC', 'value'] == 0).bool():
        return 0
    elif (img_df.loc[img_df['characteristic'] == 'ONTOPIC', 'value'] == 1).bool():
        # Image is on topic
        if (img_df.loc[img_df['characteristic'] == 'PRO', 'value'] == 1).bool():
            return 1
        elif (img_df.loc[img_df['characteristic'] == 'CON', 'value'] == 1).bool():
            return -1
    return None


def _check_models(models, model_type):
    if len(models) < 1:
        raise ValueError(f'At least one {model_type} is needed.')
    for i, model in enumerate(models):
        if not isinstance(model, model_type):
            raise ValueError(f"The given {model_type} at {i}, is not a {model_type}, but a {type(model)}.")


def start_server(t_models: List[TopicModel], a_models: List[ArgumentModel],
                 s_models: List[StanceModel], debug: bool = True, host: str = '0.0.0.0', port: int = 5000):
    log.debug('starting Flask')
    app.secret_key = '977e39540e424831d8731b8bf17f2484'
    app.debug = debug
    global topic_models, arg_models, stance_models, image_ids, valid_system
    _check_models(t_models, TopicModel)
    topic_models = t_models
    _check_models(a_models, ArgumentModel)
    arg_models = a_models
    _check_models(s_models, StanceModel)
    stance_models = s_models

    image_ids = DataEntry.get_image_ids()
    valid_system = True

    app.run(host=host, port=port, use_reloader=False)


def _parse_retrieval_system() -> RetrievalSystem:
    form_fields = ['topic_weight', 'topic_model', 'arg_weight', 'arg_model', 'stance_weight', 'stance_model']
    for field_name in form_fields:
        if field_name not in request.form.keys():
            raise ValueError(f'Missing field {field_name}')

    topic_weight = float(request.form['topic_weight'])
    topic_cls = request.form['topic_model']
    arg_weight = float(request.form['arg_weight'])
    arg_cls = request.form['arg_model']
    stance_weight = float(request.form['stance_weight'])
    stance_cls = str(request.form['stance_model'])

    for t_model in topic_models:
        if t_model.__class__.__name__ == topic_cls:
            topic_model = t_model
            break
    else:
        raise ValueError(f'Wrong topic model cls {topic_cls} '
                         f'not in {[model.__class__.__name__ for model in topic_models]}')

    for a_model in arg_models:
        if a_model.__class__.__name__ == arg_cls:
            arg_model = a_model
            break
    else:
        raise ValueError(f'Wrong argument model cls {arg_cls} '
                         f'not in {[model.__class__.__name__ for model in arg_models]}')

    for s_model in stance_models:
        if str(s_model) == stance_cls:
            stance_model = s_model
            break
    else:
        raise ValueError(f'Wrong stance model cls {stance_cls} '
                         f'not in {[str(model) for model in stance_models]}')

    return RetrievalSystem(
        prep=SpacyPreprocessor(), topic_weight=topic_weight, topic_model=topic_model,
        arg_weight=arg_weight, argument_model=arg_model, stance_weight=stance_weight, stance_model=stance_model)


@app.route('/', methods=['POST', 'GET'])
def index():
    topics = sorted(Topic.load_all(), key=lambda t: t.title)

    topic_cls = [model.__class__.__name__ for model in topic_models]
    arg_cls = [model.__class__.__name__ for model in arg_models]
    stance_cls = [str(model) for model in stance_models]

    selected_topic_cls = topic_cls[0]
    selected_arg_cls = arg_cls[0]
    selected_stance_cls = stance_cls[0]

    def standard_response():
        return render_template('index.html', results=[], topK=20, topics=topics, selected_topic=33,
                               topic_models=topic_cls, arg_models=arg_cls, stance_models=stance_cls,
                               selected_topic_cls=selected_topic_cls, selected_arg_cls=selected_arg_cls,
                               selected_stance_cls=selected_stance_cls)

    if request.method == 'POST':
        if valid_system:
            if 'retrieve_type' in request.form.keys() and 'topK' in request.form.keys():
                then = datetime.datetime.now()
                query = ''
                topic_number = None
                if request.form['retrieve_type'] == 'topic' and 'topic_field' in request.form.keys():
                    try:
                        topic_number = int(request.form['topic_field'])
                        topic = Topic.get(topic_number)
                        query = topic.title
                    except ValueError:
                        topic_number = None
                        pass
                elif request.form['retrieve_type'] == 'query':
                    if 'query' in request.form.keys():
                        query = request.form['query']

                if query == '':
                    return standard_response()

                try:
                    top_k = int(request.form['topK'])
                except ValueError:
                    top_k = 20

                try:
                    rs = _parse_retrieval_system()
                except ValueError:
                    return standard_response()
                pro_result, con_result = rs.query(query, top_k=top_k)
                now = datetime.datetime.now()

                def load_entries(ids: List[Tuple[str, float]]) -> List[Tuple[DataEntry, Union[int, None]]]:
                    result = []
                    for image_id, score in ids:
                        try:
                            entry = DataEntry.load(image_id)
                            if topic_number is not None:
                                result.append((entry, _get_image_eval(image_id, topic_number)))
                            else:
                                result.append((entry, None))
                        except ValueError:
                            pass
                    return result

                pro_images = load_entries(pro_result)
                con_images = load_entries(con_result)

                selected_topic_cls = rs.topic_model.__class__.__name__
                selected_arg_cls = rs.argument_model.__class__.__name__
                selected_stance_cls = str(rs.stance_model)

                if selected_stance_cls == 'DummyStanceModel' and selected_topic_cls != 'ArgsMeTopicModel':
                    result_sites = [('Result', '', pro_images)]
                else:
                    result_sites = [('PRO', 'text-success', pro_images), ('CON', 'text-danger', con_images)]

                return render_template('index.html',
                                       results=result_sites,
                                       search_value=query, topK=top_k,
                                       time_request=str(now - then), topics=topics, selected_topic=topic_number,
                                       topic_models=topic_cls, arg_models=arg_cls, stance_models=stance_cls,
                                       selected_topic_cls=selected_topic_cls, selected_arg_cls=selected_arg_cls,
                                       selected_stance_cls=selected_stance_cls)

    return standard_response()


@app.route('/evaluation', methods=['GET', 'POST'])
def evaluation():
    user = request.cookies.get('user_name', '')
    topics = Topic.load_all()
    selected_topic = Topic.get(1)
    image = None

    topic_done = False
    topic_len = None
    images_done = None
    done_percent = None

    if request.method == 'POST':
        if 'selected_topic' in request.form.keys() and 'user_name' in request.form.keys():
            user = request.form['user_name']
            try:
                selected_topic = Topic.get(int(request.form['selected_topic']))
            except ValueError:
                pass

        if len(user) > 0:
            if 'arg' in request.form.keys() and 'stance' in request.form.keys() \
                    and 'image_id' in request.form.keys() and 'topic_correct' in request.form.keys():
                if request.form['arg'] == 'weak':
                    arg = Argumentative.WEAK
                elif request.form['arg'] == 'strong':
                    arg = Argumentative.STRONG
                else:
                    arg = Argumentative.NONE

                if request.form['stance'] == 'pro':
                    stance = Stance.PRO
                elif request.form['stance'] == 'con':
                    stance = Stance.CON
                else:
                    stance = Stance.NEUTRAL

                if request.form['topic_correct'] == 'topic-true':
                    topic_correct = True
                else:
                    topic_correct = False

                save_eval(image_id=request.form['image_id'], user=user.replace(' ', ''), topic_correct=topic_correct,
                          topic=selected_topic.number, arg=arg, stance=stance)

    if len(user) > 0:
        image = get_image_to_eval(selected_topic)
        if image is None:
            topic_done = True
        topic_len = len(selected_topic.get_image_ids())
        images_done = 0
        for image_id in selected_topic.get_image_ids():
            if has_eval(image_id, selected_topic.number):
                images_done += 1

        done_percent = round((images_done/topic_len)*100, 2)

    resp = make_response(render_template('evaluation.html', topics=topics, selected_topic=selected_topic,
                                         user_name=user, topic_done=topic_done, image=image,
                                         images_done=images_done, topic_len=topic_len, done_percent=done_percent))
    expire = datetime.datetime.now() + datetime.timedelta(days=90)
    if len(user) > 0:
        resp.set_cookie('user_name', user, expires=expire)
    return resp


def get_abs_data_path(path):
    if not path.is_absolute():
        path = Path(os.path.abspath(__file__)).parent.parent.joinpath(path)
    return path


@app.route('/data/image/<path:image_id>')
def data_image(image_id):
    if image_id not in image_ids:
        return abort(404)
    entry = DataEntry.load(image_id)
    return send_file(get_abs_data_path(entry.webp_path))


@app.route('/data/png/<path:image_id>')
def data_image_png(image_id):
    if image_id not in image_ids:
        return abort(404)
    entry = DataEntry.load(image_id)
    if entry.png_path.exists():
        return send_file(get_abs_data_path(entry.png_path))
    return 'The png files are not in the data directory. Use the webp images.<br>' \
           f'<a href="/data/image/{image_id}">Webp Image</a>'


@app.route('/data/screenshot/<path:image_id>/<path:page_id>')
def data_snp_screenshot(image_id, page_id):
    if image_id not in image_ids:
        return abort(404)
    entry = DataEntry.load(image_id)
    for page in entry.pages:
        if page.url_hash == page_id:
            if page.snp_screenshot.exists():
                return send_file(get_abs_data_path(page.snp_screenshot))
            return 'The screenshot files are not in the data directory. Use the direct site or dom file.<br>' \
                   f'<a href="{page.url}">Website URL</a><br>' \
                   f'<a href="/data/dom/{image_id}/{page_id}">Website DOM</a>'
    return abort(404)


@app.route('/data/dom/<path:image_id>/<path:page_id>')
def data_snp_dom(image_id, page_id):
    if image_id not in image_ids:
        return abort(404)
    entry = DataEntry.load(image_id)
    for page in entry.pages:
        if page.url_hash == page_id:
            return send_file(get_abs_data_path(page.snp_dom))
    return abort(404)
