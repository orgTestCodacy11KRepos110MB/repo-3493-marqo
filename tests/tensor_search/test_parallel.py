from marqo.errors import IndexNotFoundError
import unittest
import copy
from marqo.tensor_search import parallel
import torch
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search import tensor_search
from marqo.errors import InternalError

class TestAddDocumentsPara(MarqoTestCase):
    """
    This test generates SSL warnings when running against a local Marqo because
    parallel.py turns on logging.
    """

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass
    
    def test_get_device_ids(self) -> None:
        assert parallel.get_gpu_count('cpu') == 0

        assert parallel.get_gpu_count('cuda') == torch.cuda.device_count()

        # TODO need a gpu test

    def test_get_device_ids_2(self) -> None:

        assert parallel.get_device_ids(1, 'cpu') == ['cpu']

        assert parallel.get_device_ids(2, 'cpu') == ['cpu', 'cpu']

        # TODO need a gpu test

    def test_get_processes(self) -> None:

        assert parallel.get_processes('cpu', max_processes=100) >= 1

    def test_add_documents_parallel(self) -> None:
        
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

        data = [{'text':f'something {str(i)}', '_id': str(i)} for i in range(100)]

        res = tensor_search.add_documents_orchestrator(config=self.config, index_name=self.index_name_1, docs=data,
                                                       batch_size=10, processes=2, auto_refresh=True)

        assert 'errors' in res
        assert 'index_name' in res
        assert 'processingTimeMs' in res
        assert 'items' in res
        assert len(res['items']) == len(data)

        res = tensor_search.search(config=self.config, text='something 1', index_name=self.index_name_1)

        assert res['hits'][0]['text'] == 'something 1'
    
    def test_add_documents_parallel_single_batch(self) -> None:
        
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

        data = [{'text':f'something {str(i)}', '_id': str(i)} for i in range(100)]

        res = tensor_search.add_documents_orchestrator(config=self.config, index_name=self.index_name_1, docs=data,
                                                       batch_size=200, processes=2, auto_refresh=True)

        assert 'errors' in res
        assert 'index_name' in res
        assert 'processingTimeMs' in res
        assert 'items' in res
        assert len(res['items']) == len(data)

        res = tensor_search.search(config=self.config, text='something 1', index_name=self.index_name_1)

        assert res['hits'][0]['text'] == 'something 1'

    def test_add_documents_parallel_many_batch(self) -> None:
        
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

        data = [{'text':f'something {str(i)}', '_id': str(i)} for i in range(100)]

        res = tensor_search.add_documents_orchestrator(config=self.config, index_name=self.index_name_1, docs=data,
                                                       batch_size=1, processes=2, auto_refresh=True)

        assert 'errors' in res
        assert 'index_name' in res
        assert 'processingTimeMs' in res
        assert 'items' in res
        assert len(res['items']) == len(data)

        res = tensor_search.search(config=self.config, text='something 1', index_name=self.index_name_1)

        assert res['hits'][0]['text'] == 'something 1'

    def test_consolidate_mp_results(self) -> None:

        result1 = [{'errors': False,
        'index_name': 'my-tes-index-1',
        'items': [{'_id': '123', 'result': 'updated', 'status': 200},
                {'_id': '2b049cf5-c75f-4230-8dcb-4431b8a58370',
                    'result': 'created',
                    'status': 201},
                {'_id': '456', 'result': 'updated', 'status': 200}],
        'processingTimeMs': 11805.557999999999}]

        result2 = [{'errors': False,
        'index_name': 'my-tes-index-1',
        'items': [{'_id': '1123', 'result': 'updated', 'status': 200},
                {'_id': '212b049cf5-c75f-4230-8dcb-4431b8a58370',
                    'result': 'created',
                    'status': 201},
                {'_id': '45126', 'result': 'updated', 'status': 200}],
        'processingTimeMs': 21805.557999999999}]

        result3 = [{'errors': True,
        'index_name': 'my-tes-index-1',
        'items': [{'_id': '11233', 'result': 'updated', 'status': 200},
                {'_id': 'f5-c75f-4230-8dcb-4431b8a58370',
                    'result': 'created',
                    'status': 201},
                {'_id': 'as', 'result': 'updated', 'status': 200}],
        'processingTimeMs': 1805.557999999999}]

        results = [result1, result2, result3]

        consolidated_result = parallel._consolidate_pool_results(results)

        assert consolidated_result['errors'] == True
        assert consolidated_result['index_name'] == 'my-tes-index-1'
        assert consolidated_result['processingTimeMs'] == 21805.557999999999
        assert consolidated_result['items'] == [{'_id': '123', 'result': 'updated', 'status': 200},
                {'_id': '2b049cf5-c75f-4230-8dcb-4431b8a58370',
                    'result': 'created',
                    'status': 201},
                {'_id': '456', 'result': 'updated', 'status': 200},
                {'_id': '1123', 'result': 'updated', 'status': 200},
                {'_id': '212b049cf5-c75f-4230-8dcb-4431b8a58370',
                    'result': 'created',
                    'status': 201},
                {'_id': '45126', 'result': 'updated', 'status': 200},
                {'_id': '11233', 'result': 'updated', 'status': 200},
                {'_id': 'f5-c75f-4230-8dcb-4431b8a58370',
                    'result': 'created',
                    'status': 201},
                {'_id': 'as', 'result': 'updated', 'status': 200}
                ]

    def test_consolidate_mp_results_single(self) -> None:

        result1 = [{'errors': False,
        'index_name': 'my-tes-index-1',
        'items': [{'_id': '123', 'result': 'updated', 'status': 200},
                {'_id': '2b049cf5-c75f-4230-8dcb-4431b8a58370',
                    'result': 'created',
                    'status': 201},
                {'_id': '456', 'result': 'updated', 'status': 200}],
        'processingTimeMs': 11805.557999999999}]

        consolidated_result = parallel._consolidate_pool_results(result1)

        assert consolidated_result['errors'] == False
        assert consolidated_result['index_name'] == 'my-tes-index-1'
        assert consolidated_result['processingTimeMs'] == 11805.557999999999
        assert consolidated_result['items'] == [{'_id': '123', 'result': 'updated', 'status': 200},
                {'_id': '2b049cf5-c75f-4230-8dcb-4431b8a58370',
                    'result': 'created',
                    'status': 201},
                {'_id': '456', 'result': 'updated', 'status': 200}]

    def test_consolidate_mp_results_dict(self) -> None:

        result1 = {'errors': False,
        'index_name': 'my-tes-index-1',
        'items': [{'_id': '123', 'result': 'updated', 'status': 200},
                {'_id': '2b049cf5-c75f-4230-8dcb-4431b8a58370',
                    'result': 'created',
                    'status': 201},
                {'_id': '456', 'result': 'updated', 'status': 200}],
        'processingTimeMs': 11805.557999999999}

        consolidated_result = parallel._consolidate_pool_results(result1)

        assert consolidated_result['errors'] == False
        assert consolidated_result['index_name'] == 'my-tes-index-1'
        assert consolidated_result['processingTimeMs'] == 11805.557999999999
        assert consolidated_result['items'] == [{'_id': '123', 'result': 'updated', 'status': 200},
                {'_id': '2b049cf5-c75f-4230-8dcb-4431b8a58370',
                    'result': 'created',
                    'status': 201},
                {'_id': '456', 'result': 'updated', 'status': 200}]
    
    def test_consolidate_mp_results_wrong_types(self) -> None:

        result1 = [{'errors': False,
            'index_name': 'my-tes-index-1',
            'items': [{'_id': '123', 'result': 'updated', 'status': 200},
                    {'_id': '2b049cf5-c75f-4230-8dcb-4431b8a58370',
                        'result': 'created',
                        'status': 201},
                    {'_id': '456', 'result': 'updated', 'status': 200}],
            'processingTimeMs': 11805.557999999999}]

        result2 = {'errors': False,
        'index_name': 'my-tes-index-1',
        'items': [{'_id': '1123', 'result': 'updated', 'status': 200},
                {'_id': '212b049cf5-c75f-4230-8dcb-4431b8a58370',
                    'result': 'created',
                    'status': 201},
                {'_id': '45126', 'result': 'updated', 'status': 200}],
        'processingTimeMs': 21805.557999999999}

        results = [result1, result2]

        try:
            consolidated_result = parallel._consolidate_pool_results(results)
        except InternalError as e:
            pass

    