import logging
import random
from typing import List

from .base_model import StanceModel


class RandomStanceModel(StanceModel):
    log = logging.getLogger('RdmStanceModel')

    def __init__(self):
        """
        Constructor for random stance model.
        """
        super().__init__()

    def score(self, query: List[str], doc_id: str) -> float:
        random.seed(doc_id)
        return random.choice([-1, 1])
