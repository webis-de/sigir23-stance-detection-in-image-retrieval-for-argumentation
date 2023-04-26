import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sqlitedict import SqliteDict

from config import Config
from . import html_preprocessing, sentiment_detection, image_detection
from ..data_entry import DataEntry
from ..preprocessing import SpacyPreprocessor

cfg = Config.get()


class FeatureIndex:
    log = logging.getLogger('feature_index')

    dataframe: pd.DataFrame
    text_sql: SqliteDict = None
    name: str
    _sql_file: Path

    def __init__(self, index_name: str):
        self.name = index_name
        self._sql_file = cfg.working_dir.joinpath(Path('feature_index_{}.sqlite'.format(self.name)))

    def __enter__(self):
        self.text_sql = SqliteDict(filename=self._sql_file, tablename='text', autocommit=False)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.text_sql.close()
        self.text_sql = None

    def __iter__(self):
        for i in self.dataframe.index:
            yield i

    def __len__(self) -> int:
        return len(self.dataframe)

    def _calc_doc_features(self, image_id: str) -> Tuple[List[Any], Dict[str, List[str]]]:
        self.log.debug("indexing document id = %s", image_id)

        text = html_preprocessing.run_html_preprocessing(image_id)
        if text:
            html_sentiment_score = sentiment_detection.sentiment_nltk(text)
        else:
            html_sentiment_score = 0

        image_path = DataEntry.load(image_id).png_path
        image = image_detection.read_image(image_path)

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f'PNG file for {image_id} was not read correctly.') from e

        text_analysis = image_detection.text_analysis(image_rgb)

        image_rgb_small = cv2.resize(image_rgb, (200, 200), interpolation=cv2.INTER_AREA)

        color_mood = image_detection.color_mood(image_rgb_small)
        dominant_color = image_detection.dominant_color(image_rgb_small)
        image_type = image_detection.detect_image_type(image_rgb_small)
        roi_area = image_detection.diagramms_from_image(image_rgb_small)

        text_area_str = FeatureIndex.convert_text_area_to_str(text_analysis['text_position'])

        text_dict = {}
        prep = SpacyPreprocessor()
        text_dict['image_text'] = prep.preprocess(text_analysis['text'])
        text_dict['html_text'] = prep.preprocess(text)

        id_list = [
            image_id,
            html_sentiment_score,
            text_analysis['text_len'],
            text_analysis['text_sentiment_score'],
            text_analysis['text_area_percentage'],
            text_analysis['text_area_left'],
            text_analysis['text_area_right'],
            text_analysis['text_area_top'],
            text_analysis['text_area_bottom'],
            text_area_str,
            color_mood['percentage_green'],
            color_mood['percentage_red'],
            color_mood['percentage_blue'],
            color_mood['percentage_yellow'],
            color_mood['percentage_bright'],
            color_mood['percentage_dark'],
            color_mood['average_color'][0],
            color_mood['average_color'][1],
            color_mood['average_color'][2],
            dominant_color[0],
            dominant_color[1],
            dominant_color[2],
            image_type.value,
            roi_area
        ]
        return id_list, text_dict

    def _create_df(self):
        df = pd.DataFrame(columns=[
            'image_id',
            'html_sentiment_score',
            'text_len',
            'text_sentiment_score',
            'text_area_percentage',
            'text_area_left',
            'text_area_right',
            'text_area_top',
            'text_area_bottom',
            'text_position',
            'image_percentage_green',
            'image_percentage_red',
            'image_percentage_blue',
            'image_percentage_yellow',
            'image_percentage_bright',
            'image_percentage_dark',
            'image_average_color_r',
            'image_average_color_g',
            'image_average_color_b',
            'image_dominant_color_r',
            'image_dominant_color_g',
            'image_dominant_color_b',
            'image_type',
            'image_roi_area',
        ])

        df = df.astype(dtype={
            'image_id': pd.StringDtype(),
            'html_sentiment_score': np.float,
            'text_len': np.int,
            'text_sentiment_score': np.float,
            'text_area_percentage': np.float,
            'text_area_left': np.float,
            'text_area_right': np.float,
            'text_area_top': np.float,
            'text_area_bottom': np.float,
            'text_position': pd.StringDtype(),
            'image_percentage_green': np.float,
            'image_percentage_red': np.float,
            'image_percentage_blue': np.float,
            'image_percentage_yellow': np.float,
            'image_percentage_bright': np.float,
            'image_percentage_dark': np.float,
            'image_average_color_r': np.float,
            'image_average_color_g': np.float,
            'image_average_color_b': np.float,
            'image_dominant_color_r': np.float,
            'image_dominant_color_g': np.float,
            'image_dominant_color_b': np.float,
            'image_type': np.int8,
            'image_roi_area': np.float,
        })

        df.set_index('image_id', inplace=True, verify_integrity=True)

        self.dataframe = df

    @classmethod
    def create_index(cls, name: str, max_images: int = -1, n_jobs: int = -2,
                     raise_errors: bool = False) -> 'FeatureIndex':
        """
        Create a feature index object from the stored data.
        If max_images is < 1 use all images found else stop after max_images.

        :param raise_errors: Should errors during the indexing process be suppressed?
        :param name: name of the feature index
        :param max_images: Number to determine the maximal number of images to index
        :param n_jobs: the number of processes to use, if -1 use all,
            if < -1 use max_processes+1+n_jobs, example n_jobs = -2 -> use all processors except 1.
            see joblib.parallel.Parallel
        :return: An index object
        """
        # ~122 min -> 136.9min -> 215.9min
        index = cls(name)
        index._create_df()

        if max_images != 0:
            image_ids = DataEntry.get_image_ids(max_images)

            faulty_ids = index.update(image_ids, n_jobs=n_jobs)
            index.save()
            if len(faulty_ids) > 0:
                if raise_errors:
                    raise ValueError('Not all image_ids could be used: %s', faulty_ids)
                else:
                    index.log.warning('Not all image_ids could be used: %s', faulty_ids)
        else:
            index.save()

        return index

    def _update_df(self, data):
        result = []
        with self:
            for t in data:
                result.append(t[0])
                self.text_sql[t[0][0]] = t[1]
            self.text_sql.commit()

        df = pd.DataFrame(result, columns=[
            'image_id',
            'html_sentiment_score',
            'text_len',
            'text_sentiment_score',
            'text_area_percentage',
            'text_area_left',
            'text_area_right',
            'text_area_top',
            'text_area_bottom',
            'text_position',
            'image_percentage_green',
            'image_percentage_red',
            'image_percentage_blue',
            'image_percentage_yellow',
            'image_percentage_bright',
            'image_percentage_dark',
            'image_average_color_r',
            'image_average_color_g',
            'image_average_color_b',
            'image_dominant_color_r',
            'image_dominant_color_g',
            'image_dominant_color_b',
            'image_type',
            'image_roi_area',
        ])

        df = df.astype(dtype={
            'image_id': pd.StringDtype(),
            'html_sentiment_score': np.float,
            'text_len': np.int,
            'text_sentiment_score': np.float,
            'text_area_percentage': np.float,
            'text_area_left': np.float,
            'text_area_right': np.float,
            'text_area_top': np.float,
            'text_area_bottom': np.float,
            'text_position': pd.StringDtype(),
            'image_percentage_green': np.float,
            'image_percentage_red': np.float,
            'image_percentage_blue': np.float,
            'image_percentage_yellow': np.float,
            'image_percentage_bright': np.float,
            'image_percentage_dark': np.float,
            'image_average_color_r': np.float,
            'image_average_color_g': np.float,
            'image_average_color_b': np.float,
            'image_dominant_color_r': np.float,
            'image_dominant_color_g': np.float,
            'image_dominant_color_b': np.float,
            'image_type': np.int8,
            'image_roi_area': np.float,
        })

        df.set_index('image_id', inplace=True, verify_integrity=True)

        df = pd.concat([self.dataframe, df])
        self.dataframe = df[~df.index.duplicated(keep='last')]

    def get_image_ids(self) -> List[str]:
        return self.dataframe.index.unique(0).tolist()

    def update(self, image_ids: List[str], n_jobs: int = -2, len_base_ids: int = None) -> List[Tuple[str, Any]]:

        result = []
        faulty_ids = []

        if len_base_ids is None:
            len_base_ids = len(image_ids)

        if len(image_ids) < 25:
            for image_id in image_ids:
                try:
                    image_list = self._calc_doc_features(image_id)
                    result.append(image_list)
                except Exception as a:
                    faulty_ids.append((image_id, a))
            self._update_df(result)
            self.save()
            self.log.info(f'update process: {len(self)} / {len_base_ids}')
        else:
            used_jobs = 1
            with Parallel(n_jobs=n_jobs, verbose=2) as parallel:
                dispatched_images = None
                try:
                    result = parallel(delayed(self._calc_doc_features)(image_id) for image_id in image_ids)
                except Exception:
                    dispatched_images = parallel.n_dispatched_tasks
                    used_jobs = parallel._effective_n_jobs()
                    result = parallel._output.copy()

            self._update_df(result)
            self.save()
            self.log.info(f'update process: {len(self)} / {len_base_ids}')

            if dispatched_images:
                possible_faulty = image_ids[:dispatched_images]
                missing_ids = image_ids[dispatched_images:]
                for image_list in result:
                    possible_faulty.remove(image_list[0][0])

                if len(missing_ids) > 0:
                    faulty_ids += self.update(missing_ids, n_jobs=n_jobs, len_base_ids=len_base_ids)

                faulty_ids += self.update(possible_faulty, n_jobs=int(used_jobs/2), len_base_ids=len_base_ids)

        return faulty_ids

    def save(self) -> None:
        """
        Saves the object in a file.

        :return: None
        """
        return self._save(cfg.working_dir.joinpath(Path('feature_index_{}.pkl'.format(self.name))))

    def _save(self, file: Path) -> None:
        """
        Saves the object in a file.

        :return: None
        """
        self.log.debug('save index to file')
        file.parent.mkdir(exist_ok=True, parents=True)
        self.dataframe.to_pickle(file.as_posix())
        self.log.debug('Done')

    @classmethod
    def load(cls, name: str) -> 'FeatureIndex':
        """
        Loads a feature index from a file.

        :name: name of saved index
        :return: Index object loaded from file
        :raise ValueError: if name parameter is emtpy
        """
        if name == '':
            raise ValueError(f'name parameter is empty.')
        return cls._load(cfg.working_dir.joinpath(Path('feature_index_{}.pkl'.format(name))), name)

    @classmethod
    def _load(cls, file: Path, name: str) -> 'FeatureIndex':
        """
        Loads an index from a file.

        :return: Index object loaded from file
        :raise ValueError: if file for index with given name doesn't exist
        :raise ValueError: if sql file for index with given name doesn't exist
        """

        if not file.exists():
            raise ValueError('No saved feature index for file {}'.format(file))

        index = cls(name)
        index.log.debug('Load index from file %s', file)
        index.dataframe = pd.read_pickle(file)
        if not index._sql_file.exists():
            raise ValueError(f'The sql file for the feature index {name} does not exist, file: {index._sql_file}')
        index.log.debug('Done')
        return index

    def calculate_sentiment_score_v2(self, n_jobs: int = -2) -> None:
        def calc_doc_features(image_id) -> float:
            self.log.debug("update sentiment score v2 for document id = %s", image_id)

            text = html_preprocessing.run_html_preprocessing(image_id)
            if text:
                html_sentiment_score = sentiment_detection.sentiment_nltk(text)
            else:
                html_sentiment_score = 0

            return html_sentiment_score

        with Parallel(n_jobs=n_jobs, verbose=2) as parallel:
            data = parallel(delayed(calc_doc_features)(image_id) for image_id in self.dataframe.index)

        self.dataframe = self.dataframe.assign(html_sentiment_score_v2=pd.Series(data, index=self.dataframe.index))
        self.save()

    def get_html_text(self, image_id: str) -> List[str]:
        if self.text_sql is None:
            raise ValueError('Method called outside of with block.')
        return self.text_sql[image_id]['html_text']

    def get_image_text(self, image_id: str) -> List[str]:
        if self.text_sql is None:
            raise ValueError('Method called outside of with block.')
        return self.text_sql[image_id]['image_text']

    @staticmethod
    def convert_text_area_to_str(text_area_list: Dict[int, int]) -> str:
        result = ''
        for k in text_area_list.keys():
            if text_area_list[k] == 0:
                result += '|'
            else:
                result += str(int(round(text_area_list[k]*1000, 0))) + '|'
        return result[:-1]

    @staticmethod
    def convert_text_area_from_str(text_area_str: str) -> List[int]:
        result = []
        for val in text_area_str.split('|'):
            if val == '':
                result.append(0)
            else:
                i = int(val)/1000
                result.append(i)
        return result

    def get_html_sentiment_score(self, image_id: str) -> float:
        """
        Returns the html_sentiment_score for the given image id.

        :param image_id: id of the image
        :return: html_sentiment_score for image id
        """
        return self.dataframe.loc[image_id, 'html_sentiment_score']

    def get_html_sentiment_score_v2(self, image_id: str) -> float:
        """
        Returns the html_sentiment_score_v2 for the given image id.

        :param image_id: id of the image
        :return: html_sentiment_score_v2 for image id
        """
        return self.dataframe.loc[image_id, 'html_sentiment_score_v2']

    def get_text_len(self, image_id: str) -> float:
        """
        Returns the text_len for the given image id.

        :param image_id: id of the image
        :return: text_len for image id
        """
        return self.dataframe.loc[image_id, 'text_len']

    def get_text_sentiment_score(self, image_id: str) -> float:
        """
        Returns the text_sentiment_score for the given image id.

        :param image_id: id of the image
        :return: text_sentiment_score for image id
        """
        return self.dataframe.loc[image_id, 'text_sentiment_score']

    def get_text_area_percentage(self, image_id: str) -> float:
        """
        Returns the text_area_percentage for the given image id.

        :param image_id: id of the image
        :return: text_area_percentage for image id
        """
        return self.dataframe.loc[image_id, 'text_area_percentage']

    def get_text_area_left(self, image_id: str) -> float:
        """
        Returns the text_area_left for the given image id.

        :param image_id: id of the image
        :return: text_area_left for image id
        """
        return self.dataframe.loc[image_id, 'text_area_left']

    def get_text_area_right(self, image_id: str) -> float:
        """
        Returns the text_area_right for the given image id.

        :param image_id: id of the image
        :return: text_area_right for image id
        """
        return self.dataframe.loc[image_id, 'text_area_right']

    def get_text_area_top(self, image_id: str) -> float:
        """
        Returns the text_area_top for the given image id.

        :param image_id: id of the image
        :return: text_area_top for image id
        """
        return self.dataframe.loc[image_id, 'text_area_top']

    def get_text_area_bottom(self, image_id: str) -> float:
        """
        Returns the text_area_bottom for the given image id.

        :param image_id: id of the image
        :return: text_area_bottom for image id
        """
        return self.dataframe.loc[image_id, 'text_area_bottom']

    def get_text_position(self, image_id: str) -> List[int]:
        """
        Returns the text_position for the given image id.

        :param image_id: id of the image
        :return: text_position for image id
        """
        return self.convert_text_area_from_str(self.dataframe.loc[image_id, 'text_position'])

    def get_image_percentage_green(self, image_id: str) -> float:
        """
        Returns the image_percentage_green for the given image id.

        :param image_id: id of the image
        :return: image_percentage_green for image id
        """
        return self.dataframe.loc[image_id, 'image_percentage_green']

    def get_image_percentage_red(self, image_id: str) -> float:
        """
        Returns the image_percentage_red for the given image id.

        :param image_id: id of the image
        :return: image_percentage_red for image id
        """
        return self.dataframe.loc[image_id, 'image_percentage_red']

    def get_image_percentage_blue(self, image_id: str) -> float:
        """
        Returns the image_percentage_blue for the given image id.

        :param image_id: id of the image
        :return: image_percentage_blue for image id
        """
        return self.dataframe.loc[image_id, 'image_percentage_blue']

    def get_image_percentage_yellow(self, image_id: str) -> float:
        """
        Returns the image_percentage_yellow for the given image id.

        :param image_id: id of the image
        :return: image_percentage_yellow for image id
        """
        return self.dataframe.loc[image_id, 'image_percentage_yellow']

    def get_image_percentage_bright(self, image_id: str) -> float:
        """
        Returns the image_percentage_bright for the given image id.

        :param image_id: id of the image
        :return: image_percentage_bright for image id
        """
        return self.dataframe.loc[image_id, 'image_percentage_bright']

    def get_image_percentage_dark(self, image_id: str) -> float:
        """
        Returns the image_percentage_dark for the given image id.

        :param image_id: id of the image
        :return: image_percentage_dark for image id
        """
        return self.dataframe.loc[image_id, 'image_percentage_dark']

    def get_image_average_color(self, image_id: str) -> np.ndarray:
        """
        Returns the image_average_color for the given image id. Represented as RGB in a numpy array.

        :param image_id: id of the image
        :return: image_average_color for image id
        """
        cols = ['image_average_color_r', 'image_average_color_g', 'image_average_color_b']
        return self.dataframe.loc[image_id, cols].to_numpy()

    def get_image_dominant_color(self, image_id: str) -> np.ndarray:
        """
        Returns the image_dominant_color for the given image id. Represented as RGB in a numpy array.

        :param image_id: id of the image
        :return: image_dominant_color for image id
        """
        cols = ['image_dominant_color_r', 'image_dominant_color_g', 'image_dominant_color_b']
        return self.dataframe.loc[image_id, cols].to_numpy()

    def get_image_type(self, image_id: str) -> image_detection.ImageType:
        """
        Returns the image_type for the given image id.

        :param image_id: id of the image
        :return: image_type for image id
        """
        return image_detection.ImageType(self.dataframe.loc[image_id, 'image_type'])

    def get_image_roi_area(self, image_id: str) -> float:
        """
        Returns the image_roi_area for the given image id.

        :param image_id: id of the image
        :return: image_roi_area for image id
        """
        return self.dataframe.loc[image_id, 'image_roi_area']

    def get_all_features(self, image_id: str) -> pd.Series:
        """
        Returns all images features.

        :param image_id: id of the image
        :return: all features for image_id
        """
        df = self.dataframe.loc[image_id, :]
        return df
