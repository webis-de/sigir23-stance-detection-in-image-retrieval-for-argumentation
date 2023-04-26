import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from transformers import TFCLIPModel, CLIPProcessor

from indexing import DataEntry
from .base_model import StanceModel


class ClipStanceModel(StanceModel):
    log = logging.getLogger('ClipStanceModel')

    def __init__(self):
        """
        Constructor for clip stance model.
        """
        super().__init__()
        self.model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def strip_stance(self, df: pd.DataFrame, min_k: int = 10) -> pd.DataFrame:
        stance = df['stance'].sort_values(ascending=False)

        pct = stance.pct_change()
        striped = pct.iloc[min_k:].loc[pct < -0.015]

        # diff = (stance - stance.shift(1, fill_value=0)).iloc[1:]
        # striped = diff.iloc[min_k:].loc[diff < -0.01]

        # striped = stance.iloc[min_k:].loc[stance < 0.5]
        if len(striped.index) > 0:
            image_id_strip = striped.index[0]  # diff shift
            result_ids = stance.loc[:image_id_strip].iloc[:-1]
            result = df.loc[result_ids.index, :]
        else:
            result = df

        self.log.debug(f'Striped from {len(df.index)} to {len(result.index)}')

        return result

    def query(self, query: List[str], argument_relevant: pd.DataFrame,
              top_k: int = -1, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Queries a given preprocessed query against the index using a model scoring function

        :param argument_relevant: DataFrame with data for topic and argument score
        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: Tuple of given DataFrame with an additional column for pro/con stance score.
            Frames are sorted and reduced to top_k rows
        """
        self.log.debug('start stance process for query %s', query)

        full_query = kwargs.pop('full_query')
        if full_query is None:
            full_query = ' '.join(query)

        loaded_images = [Image.open(DataEntry.load(image_id).png_path) for image_id in argument_relevant.index]

        inputs_pro = self.processor(text=full_query+' good', images=loaded_images, return_tensors="tf", padding=True)
        outputs_pro = self.model(**inputs_pro)
        result_pro = outputs_pro.logits_per_image.numpy()

        inputs_con = self.processor(text=full_query+' anti', images=loaded_images, return_tensors="tf", padding=True)
        outputs_con = self.model(**inputs_con)
        result_con = outputs_con.logits_per_image.numpy()

        argument_relevant.loc[:, 'stance'] = np.nan
        pro_scores = argument_relevant.copy()
        pro_scores.loc[:, 'stance'] = pd.Series(result_pro.flatten(), index=argument_relevant.index)
        pro_scores = self.strip_stance(pro_scores)
        con_scores = argument_relevant.copy()
        con_scores.loc[:, 'stance'] = pd.Series(result_con.flatten(), index=argument_relevant.index)
        con_scores = self.strip_stance(con_scores)

        if top_k < 0:
            top_k = len(pro_scores)
        else:
            top_k = min(len(pro_scores), top_k)

        return pro_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all'), \
            con_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all')
