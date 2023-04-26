from typing import Tuple, Dict, List, Union

import numpy as np
import pandas as pd
from collections import OrderedDict

from config import Config
from evaluation import eval_data
from indexing import data_entry


def exploratory_data_analysis() -> Tuple[Dict, Dict]:
    """
    Do Exploratory Data Analysis
    :return: Dictionary with results of EDA, Dictionary with average scores
    """
    analysis = dict()
    arguments = {"topic_name": "", "count_topic_relevance": 0, "count_images_in_topic": 0,
                 "percentage_topic_relevance": 0.0, "count_argumentative": 0, "percentage_argumentative": 0.0,
                 "count_argumentative_strong": 0, "percentage_argumentative_strong": 0.0,
                 "count_stance": 0, "percentage_stance": 0.0, "count_stance_pro": 0, "percentage_stance_pro": 0.0,
                 "count_stance_con": 0, "percentage_stance_con": 0.0, "count_stance_neutral": 0,
                 "percentage_stance_neutral": 0.0, "count_relevant_images": 0, "percentage_relevant_images": 0.0,
                 "count_relevant_images_strong": 0, "percentage_relevant_images_strong": 0.0}

    for i in data.index:
        column = data.loc[i]

        topic = column.loc["Topic"]
        topic_relevance = column.loc["Topic_correct"]
        argument = column.loc["Argumentative"]
        stance = column.loc["Stance"]

        if topic not in analysis:
            analysis.setdefault(topic, arguments.copy())
            topic_name = data_entry.Topic.get(topic)
            topic_name = topic_name.title
            analysis[topic]["topic_name"] = topic_name

        if topic_relevance:
            analysis[topic]["count_topic_relevance"] += 1

        if argument != "NONE":
            analysis[topic]["count_argumentative"] += 1

        if argument == "STRONG":
            analysis[topic]["count_argumentative_strong"] += 1

        if stance != "NEUTRAL":
            analysis[topic]["count_stance"] += 1

        if stance == "PRO":
            analysis[topic]["count_stance_pro"] += 1

        if stance == "CON":
            analysis[topic]["count_stance_con"] += 1

        if stance == "NEUTRAL":
            analysis[topic]["count_stance_neutral"] += 1

        if topic_relevance and argument != "NONE" and stance != "NEUTRAL":
            analysis[topic]["count_relevant_images"] += 1

        if topic_relevance and argument == "STRONG" and stance != "NEUTRAL":
            analysis[topic]["count_relevant_images_strong"] += 1

        analysis[topic]["count_images_in_topic"] += 1

    new_analysis = analysis.copy()

    for topic in analysis:
        if analysis[topic]["count_images_in_topic"] < 100:
            new_analysis.pop(topic)

    analysis = new_analysis

    for topic in analysis:
        analysis[topic]["percentage_topic_relevance"] = \
            analysis[topic]["count_topic_relevance"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_argumentative"] = \
            analysis[topic]["count_argumentative"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_argumentative_strong"] = \
            analysis[topic]["count_argumentative_strong"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_stance"] = \
            analysis[topic]["count_stance"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_stance_pro"] = \
            analysis[topic]["count_stance_pro"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_stance_con"] = \
            analysis[topic]["count_stance_con"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_stance_neutral"] = \
            analysis[topic]["count_stance_neutral"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_relevant_images"] = \
            analysis[topic]["count_relevant_images"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_relevant_images_strong"] = \
            analysis[topic]["count_relevant_images_strong"] / analysis[topic]["count_images_in_topic"]

    average_percentage_topic_relevance = 0
    average_percentage_argumentative = 0
    average_percentage_argumentative_strong = 0
    average_percentage_stance = 0
    average_percentage_stance_pro = 0
    average_percentage_stance_con = 0
    average_percentage_stance_neutral = 0
    average_percentage_relevant_images = 0
    average_percentage_relevant_images_strong = 0

    counter = 0
    for topic in analysis:
        counter += analysis[topic]["count_images_in_topic"]
        average_percentage_topic_relevance += analysis[topic]["count_topic_relevance"]
        average_percentage_argumentative += analysis[topic]["count_argumentative"]
        average_percentage_argumentative_strong += analysis[topic]["count_argumentative_strong"]
        average_percentage_stance += analysis[topic]["count_stance"]
        average_percentage_stance_pro += analysis[topic]["count_stance_pro"]
        average_percentage_stance_con += analysis[topic]["count_stance_con"]
        average_percentage_stance_neutral += analysis[topic]["count_stance_neutral"]
        average_percentage_relevant_images += analysis[topic]["count_relevant_images"]
        average_percentage_relevant_images_strong += analysis[topic]["count_relevant_images_strong"]

    average_percentage_topic_relevance = round(average_percentage_topic_relevance / counter, 2)
    average_percentage_argumentative = round(average_percentage_argumentative / counter, 2)
    average_percentage_argumentative_strong = round(average_percentage_argumentative_strong / counter, 2)
    average_percentage_stance = round(average_percentage_stance / counter, 2)
    average_percentage_stance_pro = round(average_percentage_stance_pro / counter, 2)
    average_percentage_stance_con = round(average_percentage_stance_con / counter, 2)
    average_percentage_stance_neutral = round(average_percentage_stance_neutral / counter, 2)
    average_percentage_relevant_images = round(average_percentage_relevant_images / counter, 2)
    average_percentage_relevant_images_strong = round(average_percentage_relevant_images_strong / counter, 2)

    eda = {"average_percentage_topic_relevance": average_percentage_topic_relevance,
           "average_percentage_argumentative": average_percentage_argumentative,
           "average_percentage_argumentative_strong": average_percentage_argumentative_strong,
           "average_percentage_stance": average_percentage_stance,
           "average_percentage_stance_pro": average_percentage_stance_pro,
           "average_percentage_stance_con": average_percentage_stance_con,
           "average_percentage_stance_neutral": average_percentage_stance_neutral,
           "average_percentage_relevant_images": average_percentage_relevant_images,
           "average_percentage_relevant_images_strong": average_percentage_relevant_images_strong}

    # sort analysis by keys
    new_analysis = dict()
    for i in sorted(list(analysis.keys())):
        new_analysis.setdefault(i, analysis[i])
    analysis = new_analysis.copy()

    return analysis, eda


def print_eda():
    """
    Print Exploratory Data Analysis
    """
    analysis, eda = exploratory_data_analysis()
    print("Average percentage of topic relevant images:", str(eda["average_percentage_topic_relevance"]))
    print("Average percentage of argumentative images:", str(eda["average_percentage_argumentative"]))
    print("Average percentage of strong argumentative images:", str(eda["average_percentage_argumentative_strong"]))
    print("Average percentage of stance not neutral images:", str(eda["average_percentage_stance"]))
    print("Average percentage of stance pro images:", str(eda["average_percentage_stance_pro"]))
    print("Average percentage of stance con images:", str(eda["average_percentage_stance_con"]))
    print("Average percentage of stance neutral images:", str(eda["average_percentage_stance_neutral"]))
    print("Average percentage of relevant images:", str(eda["average_percentage_relevant_images"]))
    print("Average percentage of relevant and strong argumentative images:",
          str(eda["average_percentage_relevant_images_strong"]))


def preprocess_string(s: str) -> str:
    """
    Preprocess String
    :param s: str
    :return: s (str)
    """
    s = s.replace("_", " ")
    s = s.title()
    return s


def create_eda_md_table(analysis: Dict, eda: Dict):
    """
    Create Markdown File with Table of EDA
    :param eda: eda Dictionary
    :param analysis: Dictionary
    """
    text = list()
    text.append("# Analysis of labeled data")
    text.append("## Analysis per Topic")
    # text.append("| Topic Number | Topic Name | Topic Relevance | Argumentative | Strong Argumentative | Stance Pro | "
    #             "Stance Con | Relevant | Strong Relevant |")
    # text.append("|---|---|---|---|---|---|---|---|---|")
    text.append("| Topic Number | Topic Name | Topic Relevance | Argumentative | Stance Pro | "
                "Stance Con | Relevant |")
    text.append("|---|---|---|---|---|---|---|")
    for i in analysis:
        column = "| " + str(i) + " "
        column += "| " + str(analysis[i]["topic_name"]) + " "
        column += "| " + str(round(analysis[i]["percentage_topic_relevance"], 2)) + " " \
                  + " (" + str(analysis[i]["count_topic_relevance"]) + ") "
        column += "| " + str(round(analysis[i]["percentage_argumentative"], 2)) + " " \
                  + " (" + str(analysis[i]["count_argumentative"]) + ") "
        # column += "| " + str(round(analysis[i]["percentage_argumentative_strong"], 2)) + " " \
        #           + " (" + str(analysis[i]["count_argumentative_strong"]) + ") "
        column += "| " + str(round(analysis[i]["percentage_stance_pro"], 2)) + " " \
                  + " (" + str(analysis[i]["count_stance_pro"]) + ") "
        column += "| " + str(round(analysis[i]["percentage_stance_con"], 2)) + " " \
                  + " (" + str(analysis[i]["count_stance_con"]) + ") "
        column += "| " + str(round(analysis[i]["percentage_relevant_images"], 2)) \
                  + " (" + str(analysis[i]["count_relevant_images"]) + ") "
        # column += "| " + str(round(analysis[i]["percentage_relevant_images_strong"], 2)) \
        #           + " (" + str(analysis[i]["count_relevant_images_strong"]) + ") "
        column += "|"
        text.append(column)

    text.append("\n")
    text.append("## Analysis Overall")
    text.append("| Category | Value |")
    text.append("|---|---|")
    for i in eda:
        column = "| " + preprocess_string(i) + " "
        column += "| " + str(eda[i]) + " "
        column += "|"
        text.append(column)

    with open('analysis_labeled_data_table_new.md', 'w') as f:
        for item in text:
            f.write("%s\n" % item)


qrel_cache: pd.DataFrame or None = None


def qrels_to_labeled_data() -> pd.DataFrame:
    if qrel_cache is not None:
        return qrel_cache
    names = ["topic", "characteristic", "web_id", "value"]
    cfg = Config.get()
    ground_truth = pd.read_csv(cfg.data_dir.joinpath("touche-task3-001-050-relevance.qrels"), sep=" ", names=names)

    cache_file = cfg.working_dir.joinpath('touche-qrels-aramis-labels.txt')
    if cache_file.exists():
        df = pd.read_csv(cache_file, sep=' ')

        df.astype(dtype={
            'image_id': pd.StringDtype(),
            'user': pd.StringDtype(),
            'Topic': np.int,
            'Topic_correct': np.bool,
            'Argumentative': pd.StringDtype(),
            'Stance': pd.StringDtype(),
        })
        df.set_index(['image_id', 'user', 'Topic'], inplace=True)
        return df

    new_data = []

    for image_id in ground_truth['web_id'].unique():
        web_df = ground_truth.loc[ground_truth['web_id'] == image_id, :]
        for topic in web_df['topic'].unique():
            topic_df = web_df.loc[web_df['topic'] == topic, ['characteristic', 'value']]
            topic_correct = topic_df.loc[topic_df['characteristic'] == 'ONTOPIC', 'value']
            pro = topic_df.loc[topic_df['characteristic'] == 'PRO', 'value']
            con = topic_df.loc[topic_df['characteristic'] == 'CON', 'value']

            if topic_correct.values[0] == 1:
                topic_correct = True
            else:
                topic_correct = False

            stance = 'NEUTRAL'
            if pro.values[0] == 1:
                stance = 'PRO'
            elif con.values[0] == 1:
                stance = 'CON'

            argumentative = 'NONE'
            if pro.values[0] == 1 or con.values[0] == 1:
                argumentative = 'STRONG'

            new_data.append({'image_id': image_id, 'user': 'touche', 'Topic': topic,
                             'Topic_correct': topic_correct, 'Argumentative': argumentative,
                             'Stance': stance})
    df = pd.DataFrame(new_data, columns=['image_id', 'user', 'Topic', 'Topic_correct', 'Argumentative', 'Stance'])

    df.astype(dtype={
        'image_id': pd.StringDtype(),
        'user': pd.StringDtype(),
        'Topic': np.int,
        'Topic_correct': np.bool,
        'Argumentative': pd.StringDtype(),
        'Stance': pd.StringDtype(),
    })
    df.set_index(['image_id', 'user', 'Topic'], inplace=True)

    df.to_csv(cache_file, sep=' ')
    return df


def labeled_data_to_qrels() -> pd.DataFrame:
    """
    Transform labeled_data to qrels format
    :return: DataFrame with qrels
    """
    qrels = pd.DataFrame(columns=["topic", "characteristic", "web_id", "value"])

    for i in range(0, len(data)):
        d = data.iloc[[i]]

        topic = d["Topic"].values[0]
        web_id = d["image_id"].values[0]

        value_topic = 0
        if d["Topic_correct"].values[0]:
            value_topic = 1

        value_pro = 0
        if d["Stance"].values[0] == "PRO":
            value_pro = 1

        value_con = 0
        if d["Stance"].values[0] == "CON":
            value_con = 1

        # ONTOPIC
        topic_dict = pd.DataFrame({"topic": [topic],
                                   "characteristic": ["ONTOPIC"],
                                   "web_id": [web_id],
                                   "value": [value_topic]})
        qrels = pd.concat([qrels, topic_dict], ignore_index=True)

        # PRO
        pro_dict = pd.DataFrame({"topic": [topic],
                                 "characteristic": ["PRO"],
                                 "web_id": [web_id],
                                 "value": [value_pro]})
        qrels = pd.concat([qrels, pro_dict], ignore_index=True)

        # CON
        con_dict = pd.DataFrame({"topic": [topic],
                                 "characteristic": ["CON"],
                                 "web_id": [web_id],
                                 "value": [value_con]})
        qrels = pd.concat([qrels, con_dict], ignore_index=True)

    return qrels


def identical_web_ids(qrels_1: pd.DataFrame, qrels_2: pd.DataFrame) -> List:
    """
    Get identical web_ids between two qrels DataFrames
    :param qrels_1: DataFrame with qrels
    :param qrels_2: DataFrame with qrels
    :return: List with identical web_ids
    """
    web_ids = list()

    for i in range(0, len(qrels_1), 3):
        id_qrels_1 = qrels_1.iloc[i]["web_id"]
        ids_qrels_2 = qrels_2["web_id"].tolist()
        if id_qrels_1 in ids_qrels_2:
            web_ids.append(id_qrels_1)

    web_ids = list(OrderedDict.fromkeys(web_ids))

    return web_ids


def compare_labels(labels_1: pd.DataFrame, labels_2: pd.DataFrame) -> Dict[Union[str, int], Dict[str, float]]:
    """
    Compare label similarity between two DataFrames with qrels
    :param labels_1: DataFrame with qrels
    :param labels_2: DataFrame with qrels
    :return: Dictionary with calculated results
    """
    results = dict()
    results.setdefault("ALL", {"ONTOPIC": 0, "PRO": 0, "CON": 0, "COUNTER": 0})
    web_ids = identical_web_ids(labels_1, labels_2
                                )
    for web_id in web_ids:
        l1 = labels_1[labels_1["web_id"] == web_id]
        l2 = labels_2[labels_2["web_id"] == web_id]

        topics_1 = list(OrderedDict.fromkeys(l1["topic"].tolist()))
        topics_2 = list(OrderedDict.fromkeys(l2["topic"].tolist()))
        topics = topics_1 + topics_2
        topics_both = list()

        for topic in topics:
            if topic in topics_1 and topic in topics_2:
                topics_both.append(topic)
                results["ALL"]["COUNTER"] += 1

        for topic in topics_both:
            if topic not in results:
                results.setdefault(topic, {"ONTOPIC": 0, "PRO": 0, "CON": 0, "COUNTER": 0})
            results[topic]["COUNTER"] += 1

            topic_l1 = l1[l1["topic"] == topic].sort_values(by=["characteristic"])
            topic_l2 = l2[l2["topic"] == topic].sort_values(by=["characteristic"])

            topic_l1_list = topic_l1["value"].tolist()
            topic_l2_list = topic_l2["value"].tolist()

            if topic_l1_list[0] == topic_l2_list[0]:
                results["ALL"]["CON"] += 1
                results[topic]["CON"] += 1

            if topic_l1_list[1] == topic_l2_list[1]:
                results["ALL"]["ONTOPIC"] += 1
                results[topic]["ONTOPIC"] += 1

            if topic_l1_list[2] == topic_l2_list[2]:
                results["ALL"]["PRO"] += 1
                results[topic]["PRO"] += 1

    for result in results:
        for characteristic in results[result]:
            if characteristic != "COUNTER":
                results[result][characteristic] = round(results[result][characteristic] / results[result]["COUNTER"], 2)

    return results


def labeled_data_comparison():
    """
    Run comparison of labeled datasets and save results as MD-File
    """
    # Transform labeled_data to qrels
    qrels_aramis = labeled_data_to_qrels()

    # load touche qrels
    names = ["topic", "characteristic", "web_id", "value"]
    qrels_touche = pd.read_csv("data/touche-task3-001-050-relevance.qrels", sep=" ", names=names)

    # Calculate similarities
    results = compare_labels(labels_1=qrels_aramis, labels_2=qrels_touche)

    # Save results as MD-File
    text = list()
    text.append("#Comparison between labeled datasets")
    text.append("* Calculated similarity scores \n")
    text.append("| Topic | ONTOPIC | PRO | CON | Images |")
    text.append("|---|---|---|---|---|")

    for i in results:
        column = "| " + str(i) + " | " + str(results[i]["ONTOPIC"]) + " | " + str(results[i]["PRO"]) + " | " + \
                 str(results[i]["CON"]) + " | " + str(results[i]["COUNTER"]) + " |"
        text.append(column)

    with open('out/labeled_data_comparison.md', 'w') as f:
        for item in text:
            f.write("%s\n" % item)


data = eval_data.get_df()
# data = qrels_to_labeled_data()
qrel_cache = qrels_to_labeled_data()
data = data.reset_index()
