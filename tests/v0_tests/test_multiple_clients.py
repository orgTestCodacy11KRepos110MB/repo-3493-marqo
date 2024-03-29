from marqo.client import Client
from marqo.neural_search import index_meta_cache
from marqo import errors
import unittest
import pprint
from tests.marqo_test import MarqoTestCase


class TestMultiClients(MarqoTestCase):

    def setUp(self) -> None:
        self.client = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except errors.MarqoApiError as s:
            pass

    def test_populate_cache_on_start(self):
        try:
            search_res_0 = self.client.index(self.index_name_1).search(
                "title about some doc")
            raise AssertionError
        except errors.MarqoApiError:
            pass
        self.client.index(self.index_name_1).add_documents([{"some title": "some field blah "}])
        search_res = self.client.index(self.index_name_1).search("title about some doc")
        assert len(search_res["hits"]) > 0
        index_meta_cache.empty_cache()
        assert len(index_meta_cache.get_cache()) == 0
        self.client = None
        # creating a client populates the cache:
        self.client = Client(**self.client_settings)
        search_res_2 = self.client.index(self.index_name_1).search("title about some doc")
        assert len(search_res_2["hits"]) > 0
