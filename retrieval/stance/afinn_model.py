import logging
from typing import List

import pandas as pd

from config import Config
from .base_model import StanceModel


class AFinnStanceModel(StanceModel):
    log = logging.getLogger('AFinnStanceModel')

    afinn_tsv: pd.DataFrame

    def __init__(self):
        """
        Constructor for random stance model.
        """
        super().__init__()
        tsv_path = Config.get().working_dir.joinpath('afinnSentimentNEW.tsv')
        df = pd.read_csv(tsv_path, sep='\t', header=0)
        df.set_index('id', drop=True, inplace=True)
        self.afinn_tsv = df

    def score(self, query: List[str], doc_id: str) -> float:
        try:
            stance = int(self.afinn_tsv.loc[doc_id, 'AFINN'])
            if stance < 0:
                return 1
            elif stance > 0:
                return -1
        except KeyError:
            self.log.warning('Found image not in AFINN scores %s', doc_id)
        return 0
