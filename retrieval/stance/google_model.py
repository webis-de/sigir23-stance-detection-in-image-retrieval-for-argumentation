import logging
import random
from typing import List

from indexing import DataEntry

from .base_model import StanceModel


class GoogleStanceModel(StanceModel):
    log = logging.getLogger('GoogleStanceModel')

    def __init__(self):
        """
        Constructor for google stance model.
        """
        super().__init__()

    def score(self, query: List[str], doc_id: str) -> float:
        entry = DataEntry.load(doc_id)
        score = 0
        best_rank = 10000
        for page in entry.pages:
            for ranking in page.rankings:
                if ranking.rank >= best_rank:
                    continue
                if ranking.query.endswith(" good"):
                    best_rank = ranking.rank
                    score = 1
                elif ranking.query.endswith(" anti"):
                    best_rank = ranking.rank
                    score = -1
        return score
