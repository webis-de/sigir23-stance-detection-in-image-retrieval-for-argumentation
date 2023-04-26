import logging
from typing import List, Tuple

from elasticsearch import Elasticsearch
from tqdm import tqdm

from .. import FeatureIndex


class ElasticSearchIndex:
    log = logging.getLogger('ElasticSearchIndex')

    def __init__(self, index_name: str):
        self.index_name = index_name
        self._connect_elastic('http://localhost:9200')

    def _connect_elastic(self, host: str, connection_retry: int = 3, connection_timeout: int = 300) -> None:
        es = Elasticsearch(hosts=host, timeout=connection_timeout)
        count = 1
        while not es.ping():
            self.log.warning(f'Could not connect to ElasticSearch at {host}, try again ({count})')
            if count > connection_retry:
                raise ConnectionError(f'Could not connect to ElasticSearch at {host}, '
                                      f'tried {count} times with timeout of {connection_timeout}')
            es = Elasticsearch(hosts=host, timeout=connection_timeout)
            count += 1
        self.es = es

    def create_index_with_features(self, findex: FeatureIndex):

        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        mapping = {
            "properties": {
                "html_text": {
                    "type": "text",
                    "similarity": "BM25"  # BM25 is the default algorithm
                },
                "image_text": {
                    "type": "text"
                },
            }
        }
        self.es.indices.create(index=self.index_name, mappings=mapping)

        with findex:
            for image_id in tqdm(findex, desc='Indexing images in ElasticSearch'):
                doc = {
                    'html_text': findex.get_html_text(image_id),
                    'image_text': findex.get_image_text(image_id),
                }
                self.es.index(index=self.index_name, id=image_id, document=doc)
        pass

    def __len__(self) -> int:
        return self.es.count(index=self.index_name)['count']

    def elastic_query(self, query: str, top_k: int, image_ids: List[str] = None) -> List[Tuple[str, float]]:
        query_mapping = {
            "bool": {
                "should": [
                    {"match": {"html_text": query}},
                    {"match": {"image_text": {"query": query, "boost": 3}}}
                ]
            }
        }
        if image_ids is not None:
            query_mapping['bool']['filter'] = {
                'ids': {
                    'values': image_ids
                }
            }
        resp = self.es.search(index=self.index_name, size=top_k, query=query_mapping)
        result = []
        for doc in resp.get('hits').get('hits'):
            image_id = doc.get('_id')
            score = doc.get('_score')
            result.append((image_id, score))

        return result
