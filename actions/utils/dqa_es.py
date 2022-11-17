"""
Name : cqa_es.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/10/28 8:30
Desc:
"""
import numpy as np
from elasticsearch import Elasticsearch, helpers, NotFoundError
import tqdm
import requests
from typing import List, Dict
from loguru import logger
import re
import pymysql
import time
from config import ES_DOCKER_IP

# 打开数据库连接，注意passwd只接收str
db = pymysql.connect(host="127.0.0.1", port=3306, user="root", passwd='123456', db="official_document", charset='utf8')
# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()


class ElasticSearchBM25(object):
    def __init__(
            self,
            index_name: str,
            reindexing: bool = True,
            port_http: int = 9200,
            host: str = None,
            timeout: int = 100,
            max_waiting: int = 100,
    ):
        self.pid = None
        if host is not None:
            self._wait_and_check(host, port_http, max_waiting)
            logger.info(f"Successfully reached out to ES service at {host}:{port_http}")
        else:
            host = f"http://{ES_DOCKER_IP}"
            if self._check_service_running(host, port_http):
                logger.info(
                    f"Successfully reached out to ES service at {host}:{port_http}"
                )
        es = Elasticsearch(
            [
                {"host": f"{ES_DOCKER_IP}", "port": port_http, "scheme": "http"},
            ],
            timeout=timeout,
        )
        logger.info(
            f"Successfully built connection to ES service at {host}:{port_http}"
        )
        self.es = es
        if es.indices.exists(index=index_name):
            if reindexing:
                logger.info(
                    f"Index {index_name} found and it will be indexed again since reindexing=True"
                )
                es.indices.delete(index=index_name)
                logger.info(f"delete index and now do indexing again")
                self._index_corpus(index_name)
        else:
            logger.info(f"No index found and now do indexing")
            self._index_corpus(index_name)
        self.index_name = index_name
        logger.info("All set up.")

    @staticmethod
    def _check_service_running(host, port) -> bool:
        """
        Check whether the ES service is reachable.
        :param host: The host address.
        :param port: The HTTP port.
        :return: Whether the ES service is reachable.
        """
        try:
            return requests.get(f"{host}:{port}").status_code == 200
        except:
            return False

    def _wait_and_check(self, host, port, max_waiting):
        logger.info(
            f"Waiting for the ES service to be well started. Maximum time waiting: {max_waiting}s"
        )
        timeout = True
        for _ in tqdm.trange(max_waiting):
            if self._check_service_running(host, port):
                timeout = False
                break
            time.sleep(1)
        assert timeout is False, (
                "Timeout to start the ES docker container or connect to the ES service, "
                + "please increase max_waiting or check the idling ES services "
                + "(starting multiple ES instances from ES executable is not allowed)"
        )

    def _index_corpus(self, index_name):
        """
        Index the corpus.
        :param index_name: The name of the target ES index.
        """
        es_index = {
            "mappings": {
                "properties": {
                    "document": {"type": "text"},
                }
            }
        }
        self.es.indices.create(index=index_name, body=es_index, ignore=[400])
        documents = []
        databases = ['education',
                     'research',
                     'administration',
                     'meeting',
                     'student_work',
                     'life', ]
        dids = []
        for database in databases:
            try:
                sql = f"SELECT pid, title, content, src, news_time from {database}"
                cursor.execute(sql)
                fetch_data = cursor.fetchall()
                for p, t, c, s, n in fetch_data:
                    dids.append(database + '_' + str(p))
                    t = re.sub("\t", "", t)
                    c = re.sub("\t", "", c)
                    s = re.sub("\t", "", s)
                    n = re.sub("\t", "", str(n))
                    # documents.append(t + '\t' + c + '\t' + s)
                    documents.append(t + '\t' + s + '\t' + n)
            except Exception as e:
                logger.error(e)
                # 发生错误时回滚
                db.rollback()

        ndocuments = len(documents)
        chunk_size = 500
        pbar = tqdm.trange(0, ndocuments, chunk_size)
        for begin in pbar:
            did_chunk = dids[begin: begin + chunk_size]
            document_chunk = documents[begin: begin + chunk_size]
            bulk_data = [
                {
                    "_index": index_name,
                    "_id": did,
                    "_source": {
                        "document": documnt,
                    },
                }
                for did, documnt in zip(did_chunk, document_chunk)
            ]
            helpers.bulk(self.es, bulk_data)
        self.es.indices.refresh(
            index=index_name
        )  # important!!! otherwise es might return nothing!!!
        logger.info(f"Indexing work done: {ndocuments} documents indexed")

    def query(self, query: str, topk, return_scores=False) -> List:
        """
        Search for a given query.
        :param query: The query text.
        :param topk: Specifying how many top documents to return. Should less than 10000.
        :param return_scores: Whether to return the scores.
        :return: Ranked documents, a mapping from IDs to the documents (and also the scores, a mapping from IDs to scores).
        """
        assert topk <= 10000, "`topk` is too large!"
        result = self.es.search(
            index=self.index_name,
            size=min(topk, 10000),
            body={"query": {"match": {"document": query}}},
        )
        hits = result["hits"]["hits"]
        documents_ranked = {hit["_id"]: hit["_source"]["document"] for hit in hits}
        if return_scores:
            scores_ranked = {hit["_id"]: hit["_score"] for hit in hits}
            return [documents_ranked, scores_ranked]
        else:
            return [documents_ranked]

    def score(
            self, query: str, document_ids: List[int], max_ntries=60
    ) -> Dict[str, str]:
        """
        Scoring a query against the given documents (IDs).
        :param query: The query text.
        :param document_ids: The document IDs.
        :param max_ntries: Maximum time (in seconds) for trying.
        :return: The mapping from IDs to scores.
        """
        for i in range(max_ntries):
            try:
                scores = {}
                for document_id in document_ids:
                    result = self.es.explain(
                        index=self.index_name,
                        id=document_id,
                        body={"query": {"match": {"document": query}}},
                    )
                    scores[document_id] = result["explanation"]["value"]
                return scores
            except NotFoundError as e:
                if i == max_ntries:
                    raise e
                logger.info(f"NotFoundError, now re-trying ({i + 1}/{max_ntries}).")
                time.sleep(1)

    def delete_index(self):
        """
        Delete the used index.
        """
        if self.es.indices.exists(index=self.index_name):
            logger.info(
                f'Delete "{self.index_name}": {self.es.indices.delete(self.index_name)}'
            )
        else:
            logger.warning(f'Index "{self.index_name}" does not exist!')


if __name__ == '__main__':
    # index name不能大写！！！！
    es = ElasticSearchBM25(index_name='dqa', reindexing=True)
