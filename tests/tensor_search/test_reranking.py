import math
import pprint
from unittest import mock
from marqo.tensor_search.enums import TensorField, SearchMethod, EnvVars
from marqo.errors import (
    MarqoApiError, MarqoError, IndexNotFoundError, InvalidArgError,
    InvalidFieldNameError, IllegalRequestedDocCount
)
from marqo.tensor_search import tensor_search, constants, index_meta_cache
import copy
from tests.marqo_test import MarqoTestCase
import requests
import random


class TestReranking(MarqoTestCase):

    def setUp(self) -> None:
        self.index_name_1 = "my-test-index-1"
        self._delete_test_indices()

        def ADD_API_KEY():

            raise Exception("REPLACE THIS WITH YOUR OPENAI API KEY")

        self.openai_api_key = ADD_API_KEY()

    def _delete_test_indices(self, indices=None):
        if indices is None or not indices:
            ix_to_delete = [self.index_name_1]
        else:
            ix_to_delete = indices
        for ix_name in ix_to_delete:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix_name)
            except IndexNotFoundError as s:
                pass

    def tearDown(self) -> None:
        self._delete_test_indices()

    def test_gpt_reranking_qa(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"Summary": "The Moon Hawks is a space ball team.", "Title": "Top Space Ball Teams",
                 "_id": "5678"},
                {"Summary": "Roolio Moovlen was the captain of the Moon Hawks", "Title": "Legendary Space Ball players",
                 "_id": "1234"},
            ], auto_refresh=True)

        search_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="Who is the top space ball player?",
            return_doc_ids=True, result_count=10, reranker="openai/gpt3-qa",
            searchable_attributes=["Summary", "Title"],
            reranker_properties={
                "api_key": self.openai_api_key,
                "pre_summarise": True
            }
        )
        pprint.pprint(search_res)

    def test_gpt_reranking_freeform(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"Summary": "The Moon Hawks is a space ball team.", "Title": "Top Space Ball Teams",
                 "_id": "5678"},
                {"Summary": "Roolio Moovlen was the captain of the Moon Hawks", "Title": "Legendary Space Ball players",
                 "_id": "1234"},
            ], auto_refresh=True)

        search_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="Who is the top space ball player?",
            return_doc_ids=True, result_count=10, reranker="openai/gpt3-freeform",
            searchable_attributes=["Summary", "Title"],
            reranker_properties={
                "api_key": self.openai_api_key,
                "instruction": "Sort main entities in the sources alphabetically. "
                               "Add the title 'Entities' before the list of entities. "
                               "Separate entities via commas. Answer:",
            }
        )
        pprint.pprint(search_res)

    def test_gpt_reranking_summarise(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"Summary": "The Moon Hawks is a space ball team.", "Title": "Top Space Ball Teams",
                 "_id": "5678"},
                {"Summary": "Roolio Moovlen was the captain of the Moon Hawks", "Title": "Legendary Space Ball players",
                 "_id": "1234"},
            ], auto_refresh=True)

        search_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="Who is the top space ball player?",
            return_doc_ids=True, result_count=10, reranker="openai/gpt3-summarise",
            searchable_attributes=["Summary", "Title"],
            reranker_properties={
                "api_key": self.openai_api_key,
                "max_tokens": 1000,
                "pre_summarise": True
            }
        )
        pprint.pprint(search_res)

    def test_gpt_reranking_reorder(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"Summary": "The Moon Hawks is a space ball team.", "Title": "Top Space Ball Teams",
                 "_id": "5678"},
                {"Summary": "Roolio Moovlen was the captain of the Moon Hawks", "Title": "Legendary Space Ball players",
                 "_id": "1234"},
            ], auto_refresh=True)

        search_res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text="Who is the top space ball player?",
            return_doc_ids=True, result_count=10, reranker="openai/gpt3-reorder",
            searchable_attributes=["Summary", "Title"],
            reranker_properties={
                "api_key": self.openai_api_key,
                "pre_summarise": True
            }
        )
        pprint.pprint(search_res)

    def test_gpt_reranking_no_searchable_attribs(self):
        """Consider moving to s2_inference tests"""

    def test_gpt_reranking_big_docs(self):
        """"""

    def test_gpt_reranking_lots_of_docs(self):
        """"""