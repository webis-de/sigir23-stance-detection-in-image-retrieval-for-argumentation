from typing import Dict, Union, List
import logging
from pathlib import Path
from collections import OrderedDict
import pandas as pd

from indexing import Topic
from retrieval import RetrievalSystem
from config import Config
from .analysis_labeled_data import labeled_data_to_qrels

log = logging.getLogger("Evaluation")
cfg = Config.get()


def calculate_precision_scores(run_path: Path, k: int = 10) -> Dict[Union[str, int], Dict[str, float]]:
    """
    Calculate precision scores from run.txt
    :param run_path: Path of run
    :param k: number of images for topic and side (PRO / CON)
    :return: Dictionary with results
    """
    # load touche qrels
    names = ["topic", "characteristic", "web_id", "value"]
    ground_truth = pd.read_csv(cfg.data_dir.joinpath("touche-task3-001-050-relevance.qrels"), sep=" ", names=names)
    # ground_truth = labeled_data_to_qrels()  # TODO

    # load run
    names = ["topic", "stance", "image_id", "rank", "score", "method"]
    run = pd.read_csv(run_path, sep=" ", names=names)

    results = dict()
    topics = list(OrderedDict.fromkeys(run["topic"].tolist()))

    for topic in topics:
        results.setdefault(topic, {"topic": 0, "argument": 0, "stance": 0, "stance_pro": 0, "stance_con": 0})
        topic_run = run[run["topic"] == topic].sort_values(by=["stance"])

        for i in topic_run.index:
            image = topic_run.loc[i]

            topic = image["topic"]
            web_id = image["image_id"]
            stance = image["stance"]

            gt = ground_truth[(ground_truth.topic == topic) & (ground_truth.web_id == web_id)].sort_values(
                by=["characteristic"])
            gt_list = gt["value"].tolist()

            if len(gt_list) == 3:
                if gt_list[1] == 1:
                    results[topic]["topic"] += 1

                if gt_list[0] == 1 or gt_list[2] == 1:
                    results[topic]["argument"] += 1

                if stance == "PRO":
                    if gt_list[2] == 1:
                        results[topic]["stance"] += 1
                        results[topic]["stance_pro"] += 1
                elif stance == "CON":
                    if gt_list[0] == 1:
                        results[topic]["stance"] += 1
                        results[topic]["stance_con"] += 1

    amount_topics = len(results)
    results.setdefault("all", {"topic": 0, "argument": 0, "stance": 0, "stance_con": 0, "stance_pro": 0})
    for result in results:
        for score in results[result]:
            if result != "all":
                results["all"][score] += results[result][score]

                if results[result][score] != 0:
                    if score == "stance_con" or score == "stance_pro":
                        results[result][score] = round(results[result][score] / k, 3)
                    else:
                        results[result][score] = round(results[result][score] / (k * 2), 3)

            else:
                if score == "stance_con" or score == "stance_pro":
                    results[result][score] = round(results[result][score] / (k * amount_topics), 3)
                else:
                    results[result][score] = round(results[result][score] / (k * 2 * amount_topics), 3)
    return results


def calculate_min_max_precision_scores(run_path: Path, k: int = 10) -> Dict[Union[str, int], Dict[str, float]]:
    """
    Calculate minimum and maximum precision scores from run.txt
    :param run_path: Path of run
    :param k: number of images for topic and side (PRO / CON)
    :return: Dictionary with results
    """
    # load touche qrels
    names = ["topic", "characteristic", "web_id", "value"]
    ground_truth = pd.read_csv(cfg.data_dir.joinpath("touche-task3-001-050-relevance.qrels"), sep=" ", names=names)

    # load run
    names = ["topic", "stance", "image_id", "rank", "score", "method"]
    run = pd.read_csv(run_path, sep=" ", names=names)

    results = dict()
    topics = list(OrderedDict.fromkeys(run["topic"].tolist()))

    for topic in topics:
        results.setdefault(topic, {"topic_min": 0, "topic_max": 0, "argument_min": 0, "argument_max": 0,
                                   "stance_min": 0, "stance_max": 0, "stance_min_con": 0, "stance_max_con": 0,
                                   "stance_min_pro": 0, "stance_max_pro": 0})
        topic_run = run[run["topic"] == topic].sort_values(by=["stance"])

        for i in topic_run.index:
            image = topic_run.loc[i]

            topic = image["topic"]
            web_id = image["image_id"]
            stance = image["stance"]

            gt = ground_truth[(ground_truth.topic == topic) & (ground_truth.web_id == web_id)].sort_values(
                by=["characteristic"])
            gt_list = gt["value"].tolist()

            if len(gt_list) == 3:

                if gt_list[1] == 1:
                    results[topic]["topic_min"] += 1
                    results[topic]["topic_max"] += 1

                if gt_list[0] == 1 or gt_list[2] == 1:
                    results[topic]["argument_min"] += 1
                    results[topic]["argument_max"] += 1

                if stance == "PRO":
                    if gt_list[2] == 1:
                        results[topic]["stance_min"] += 1
                        results[topic]["stance_max"] += 1
                        results[topic]["stance_min_pro"] += 1
                        results[topic]["stance_max_pro"] += 1
                else:
                    if gt_list[0] == 1:
                        results[topic]["stance_min"] += 1
                        results[topic]["stance_max"] += 1
                        results[topic]["stance_min_con"] += 1
                        results[topic]["stance_max_con"] += 1
            else:
                results[topic]["topic_max"] += 1
                results[topic]["argument_max"] += 1
                results[topic]["stance_max"] += 1
                results[topic]["stance_max_pro"] += 1
                results[topic]["stance_max_con"] += 1

    results.setdefault("all", {"topic_min": 0, "topic_max": 0, "argument_min": 0, "argument_max": 0, "stance_min": 0,
                               "stance_max": 0, "stance_min_con": 0, "stance_max_con": 0, "stance_min_pro": 0,
                               "stance_max_pro": 0})
    for result in results:
        for score in results[result]:
            if result != "all":
                stance_list = ["stance_min_con", "stance_max_con", "stance_min_pro", "stance_max_pro"]
                if score in stance_list:
                    results[result][score] = round((results[result][score] / (k * 2)) * 2, 3)
                else:
                    results[result][score] = round(results[result][score] / (k * 2), 3)
                results["all"][score] += results[result][score]

    for score in results["all"]:
        results["all"][score] = round(results["all"][score] / len(topics), 3)

    return results


def create_md_file(results: Dict[Union[str, int], Dict[str, float]], path: Path, min_max_scores: bool = False):
    """
    Save precision scores as MD-File
    :param results: Dictionary with created results
    :param path: Path to specify where to save MD-File
    :param min_max_scores: Set True, if MD-File for min, max precision scores should be created
    """
    text = list()
    text.append("# Precision Scores")

    if not min_max_scores:
        text.append("| Topic | Topic-Relevance | Argumentativeness | Stance | Stance (Con) | Stance (Pro) |")
        text.append("|---|---|---|---|---|---|")

        string = "| Overall | " + str(results["all"]["topic"]) + " | " + str(results["all"]["argument"]) + " | " + \
                 str(results["all"]["stance"]) + " | " + str(results["all"]["stance_con"]) + " | " + \
                 str(results["all"]["stance_pro"]) + " |"
        text.append(string)

        for i in results:
            if i != "all":
                string = "| " + str(i) + " | " + str(results[i]["topic"]) + " | " + str(results[i]["argument"]) + \
                         " | " + str(results[i]["stance"]) + " | " + str(results[i]["stance_con"]) + " | " + \
                         str(results[i]["stance_pro"]) + " |"
                text.append(string)

        path = path.joinpath("final_results.md")

    else:
        text.append("| Topic | Topic-Relevance (Min) | Topic-Relevance (Max) | Argumentativeness (Min) | "
                    "Argumentativeness (Max) | Stance (Min) | Stance (Max) | Stance Con (Min) | Stance Con (Max) | "
                    "Stance Pro (Min) | Stance Pro (Max) |")
        text.append("|---|---|---|---|---|---|---|---|---|---|---|")

        string = "| Overall | " + str(results["all"]["topic_min"]) + " | " + str(results["all"]["topic_max"]) + \
                 " | " + str(results["all"]["argument_min"]) + " | " + str(results["all"]["argument_max"]) + " | " + \
                 str(results["all"]["stance_min"]) + " | " + str(results["all"]["stance_max"]) + " | " + \
                 str(results["all"]["stance_min_con"]) + " | " + str(results["all"]["stance_max_con"]) + " | " + \
                 str(results["all"]["stance_min_pro"]) + " | " + str(results["all"]["stance_max_pro"]) + " |"
        text.append(string)

        for i in results:
            if i != "all":
                string = "| " + str(i) + " | " + str(results[i]["topic_min"]) + " | " + str(results[i]["topic_max"]) + \
                         " | " + str(results[i]["argument_min"]) + " | " + str(results[i]["argument_max"]) + " | " + \
                         str(results[i]["stance_min"]) + " | " + str(results[i]["stance_max"]) + " | " + \
                         str(results[i]["stance_min_con"]) + " | " + str(results[i]["stance_max_con"]) + " | " + \
                         str(results[i]["stance_min_pro"]) + " | " + str(results[i]["stance_max_con"]) + " |"
                text.append(string)

        path = path.joinpath("min_max_results.md")

    # save results as MD-file
    try:
        with open(path, 'w', encoding="utf-8") as f:
            for item in text:
                f.write("%s\n" % item)
        log.info(f'results saved to {path}')
    except FileNotFoundError:
        if not min_max_scores:
            with open(cfg.working_dir.joinpath("final_results.md"), 'w', encoding="utf-8") as f:
                for item in text:
                    f.write("%s\n" % item)
            log.info("FileNotFoundError: final_results.md saved at working dir")
        else:
            with open(cfg.working_dir.joinpath("min_max_results.md"), 'w', encoding="utf-8") as f:
                for item in text:
                    f.write("%s\n" % item)
            log.info("FileNotFoundError: min_max_results.md saved at working dir")


def get_save_path(method_tag: str) -> Path:
    # Preprocess method-tag
    split = method_tag.split('#')
    # TopicModel
    topic_split = split[1].split(':')
    topic_model = topic_split[1]
    # ArgumentModel
    arg_split = split[2].split(':')
    if arg_split[1][:3] == "NN-":
        arg_model = arg_split[1][3:]
    else:
        arg_model = arg_split[1]
    # StanceModel
    stance_split = split[3].split(':')
    if stance_split[1][:3] == "NN-":
        stance_model = stance_split[1][3:]
    else:
        stance_model = stance_split[1]

    # define path
    save_path = cfg.output_dir.joinpath(topic_model).joinpath(arg_model).joinpath(stance_model)
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def evaluation(method_tag: str, topics: List[int] = None):
    """
    Run evaluation for specified model configuration
    :param topics: List of topic id to evaluate
    :param method_tag: method-tag with model information
    """
    # RetrievalSystem
    rs = RetrievalSystem.parse_method_tag(method_tag=method_tag, only_eval_images=True)

    save_path = get_save_path(method_tag)

    if topics is None:
        topics = [topic.number for topic in Topic.load_all()]

    # run scoring
    rs.qrel_scoring(method_tag=method_tag, save_path=save_path.joinpath(f"run_{len(topics)}.txt"), topics=topics)

    # calculate precision scores
    results_final = calculate_precision_scores(run_path=save_path.joinpath("run.txt"))
    results_min_max = calculate_min_max_precision_scores(run_path=save_path.joinpath("run.txt"))

    # save scores as MD-File
    create_md_file(results=results_final, path=save_path, min_max_scores=False)
    create_md_file(results=results_min_max, path=save_path, min_max_scores=True)
