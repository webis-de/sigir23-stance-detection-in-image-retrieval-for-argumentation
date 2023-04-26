import logging
from typing import List

import pandas as pd

from config import Config
from .base_model import StanceModel


class BertStanceModel(StanceModel):
    log = logging.getLogger('BertStanceModel')

    bert_tsv: pd.DataFrame

    def __init__(self):
        """
        Constructor for random stance model.
        """
        super().__init__()
        tsv_path = Config.get().working_dir.joinpath('htmlTitlesStanceNEW.tsv')
        df = pd.read_csv(tsv_path, sep='\t', header=0)
        df.set_index('id', drop=True, inplace=True)
        self.bert_tsv = df

    def score(self, query: List[str], doc_id: str) -> float:
        try:
            stance = str(self.bert_tsv.loc[doc_id, 'stance'])
            if stance.startswith('P'):
                return 1
            elif stance.startswith('N'):
                return -1
        except KeyError:
            self.log.warning('Found image not in BERT scores %s', doc_id)
        return 0
