from marqo.neural_search import enums, validation
import unittest
import copy
from marqo.errors import MarqoError


class TestValidation(unittest.TestCase):

    def test_validate_chunk_plus_name(self):
        try:
            validation.validate_field_name("__chunks.__field_name")
            raise AssertionError
        except MarqoError as s:
            assert "protected field name" in str(s)

    def test_nesting_attempt(self):
        try:
            validation.validate_field_name("some_object.__field_name")
            raise AssertionError
        except MarqoError as s:
            assert "Illegal character '.'" in str(s)

    def test_validate_field_name_good(self):
        assert "some random fieldname" == validation.validate_field_name("some random fieldname")

    def test_validate_field_name_good_2(self):
        assert "abc__field_name" == validation.validate_field_name("abc__field_name")

    def test_validate_field_name_empty(self):
        try:
            validation.validate_field_name("")
            raise AssertionError
        except MarqoError as s:
            assert "empty" in str(s)

    def test_validate_field_name_none(self):
        try:
            validation.validate_field_name(None)
            raise AssertionError
        except MarqoError as s:
            assert "must be str" in str(s)

    def test_validate_field_name_other(self):
        try:
            validation.validate_field_name(123)
            raise AssertionError
        except MarqoError as s:
            assert "must be str" in str(s)

    def test_validate_field_name_protected(self):
        try:
            validation.validate_field_name("__field_name")
            raise AssertionError
        except MarqoError as s:
            assert "protected field" in str(s)

    def test_validate_field_name_vector_prefix(self):
        try:
            validation.validate_field_name("__vector_")
            raise AssertionError
        except MarqoError as s:
            assert "protected prefix" in str(s)

    def test_validate_field_name_vector_prefix_2(self):
        try:
            validation.validate_field_name("__vector_abc")
            raise AssertionError
        except MarqoError as s:
            assert "protected prefix" in str(s)

    def test_validate_doc_empty(self):
        try:
            validation.validate_doc({})
            raise AssertionError
        except MarqoError as s:
            assert "empty" in str(s)

    def test_validate_vector_name(self):
        good_name = "__vector_Title 1"
        assert good_name == validation.validate_vector_name(good_name)

    def test_validate_vector_name_2(self):
        """should only try removing the first prefix"""
        good_name = "__vector___vector_1"
        assert good_name == validation.validate_vector_name(good_name)

    def test_validate_vector_name_only_prefix(self):
        bad_vec = "__vector_"
        try:
            validation.validate_vector_name(bad_vec)
            raise AssertionError
        except MarqoError as s:
            assert "empty" in str(s)

    def test_validate_vector_empty(self):
        bad_vec = ""
        try:
            validation.validate_vector_name(bad_vec)
            raise AssertionError
        except MarqoError as s:
            assert "empty" in str(s)

    def test_validate_vector_int(self):
        bad_vec = 123
        try:
            validation.validate_vector_name(bad_vec)
            raise AssertionError
        except MarqoError as s:
            assert 'must be str' in str(s)

        bad_vec_2 = ["efg"]
        try:
            validation.validate_vector_name(bad_vec_2)
            raise AssertionError
        except MarqoError as s:
            assert 'must be str' in str(s)

    def test_validate_vector_no_prefix(self):
        bad_vec = "some bad title"
        try:
            validation.validate_vector_name(bad_vec)
            raise AssertionError
        except MarqoError as s:
            assert 'vectors must begin' in str(s)

    def test_validate_vector_name_protected_field(self):
        """the vector name without the prefix can't be the name of a protected field"""
        bad_vec = "__vector___chunk_ids"
        try:
            validation.validate_vector_name(bad_vec)
            raise AssertionError
        except MarqoError as s:
            assert 'protected name' in str(s)

    def test_validate_vector_name_id_field(self):
        bad_vec = "__vector__id"
        try:
            validation.validate_vector_name(bad_vec)
            raise AssertionError
        except MarqoError as s:
            assert 'protected name' in str(s)

    def test_validate_field_name_highlight(self):
        bad_name = "_highlights"
        try:
            validation.validate_field_name(bad_name)
            raise AssertionError
        except MarqoError as s:
            assert 'protected field' in str(s)