"""Test SSSEA."""

import unittest
from io import StringIO
from textwrap import dedent

from curies.vocabulary import xsd_integer, xsd_string

from sssea.api import (
    DEFAULT_ANNOTATION_KEY,
    BinaryAnnotation,
    BinaryAnnotationMetadata,
    CategoricalAnnotation,
    CategoricalAnnotationMetadata,
    FeatureColumn,
    IdentifierEntityParser,
    Metadata,
    MissingColumnError,
    Record,
    read,
)


class TestSSSEA(unittest.TestCase):
    """Test case for SSSEA."""

    @staticmethod
    def _read(contents: str, *, metadata: Metadata | None = None) -> list[Record]:
        rv = StringIO(contents)
        rv.seek(0)
        return read(rv, metadata=metadata)

    def _read_one(self, contents: str, *, metadata: Metadata | None = None) -> Record:
        records = self._read(contents, metadata=metadata)
        self.assertEqual(1, len(records))
        record = records[0]
        return record

    def test_raise_on_missing_column(self) -> None:
        """Test raising on missing required column."""
        document = dedent("""\
            entity	curator	date
            pubmed:30380085	orcid:0009-0009-3402-8013	2025-10-07
        """)
        with self.assertRaises(ValueError):
            self._read_one(document)

    def test_raise_on_missing_value(self) -> None:
        """Test raising on missing required key."""
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            pubmed:30380085		orcid:0009-0009-3402-8013	2025-10-07
        """)
        with self.assertRaises(ValueError):
            self._read_one(document)

    def test_raise_on_bad_binary(self) -> None:
        """Test a binary annotation."""
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            pubmed:30380085	invalid-binary-value	orcid:0009-0009-3402-8013	2025-10-07
        """)
        with self.assertRaises(ValueError):
            self._read_one(document)

    def test_raise_on_duplicate_key(self) -> None:
        """Test a raising when multiple annotations have the same key."""
        metadata = Metadata(
            categories=[
                BinaryAnnotationMetadata(key="key", nullable=False),
                CategoricalAnnotationMetadata(key="key", nullable=False),
            ]
        )
        with self.assertRaises(ValueError):
            self._read_one("<empty document>", metadata=metadata)

    def test_bad_entity_configuration(self) -> None:
        """Test raising when the expected entity column is not present."""
        document = dedent(f"""\
            not-entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            pubmed:30380085	true	orcid:0009-0009-3402-8013	2025-10-07
            pubmed:30380086	false	orcid:0009-0009-3402-8013	2025-10-07
        """)
        with self.assertRaises(ValueError):
            self._read(document)

    def test_skipping_bad_entity(self) -> None:
        """Test raising when the expected entity column is not present."""
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            	true	orcid:0009-0009-3402-8013	2025-10-07
        """)  # noqa:E101
        records = self._read(document)
        self.assertEqual(0, len(records))

    def test_empty(self) -> None:
        """Test a binary annotation."""
        metadata = Metadata(
            categories=[BinaryAnnotationMetadata(key=DEFAULT_ANNOTATION_KEY, nullable=True)]
        )
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            pubmed:30380085		orcid:0009-0009-3402-8013	2025-10-07
        """)
        records = self._read(document, metadata=metadata)
        self.assertEqual(0, len(records))

    def test_binary(self) -> None:
        """Test a binary annotation."""
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            pubmed:30380085	true	orcid:0009-0009-3402-8013	2025-10-07
            pubmed:30380086	false	orcid:0009-0009-3402-8013	2025-10-07
        """)
        records = self._read(document)
        self.assertEqual(2, len(records))

        true_record = records[0]
        self.assertEqual("pubmed:30380085", true_record.reference.curie)
        self.assertEqual({DEFAULT_ANNOTATION_KEY}, set(true_record.annotations))
        annotation = true_record.annotations[DEFAULT_ANNOTATION_KEY]
        self.assertIsInstance(annotation, BinaryAnnotation)
        self.assertTrue(annotation.value)

        false_record = records[1]
        self.assertEqual("pubmed:30380086", false_record.reference.curie)
        self.assertEqual({DEFAULT_ANNOTATION_KEY}, set(false_record.annotations))
        annotation = false_record.annotations[DEFAULT_ANNOTATION_KEY]
        self.assertIsInstance(annotation, BinaryAnnotation)
        self.assertFalse(annotation.value)

    def test_binary_custom_entity(self) -> None:
        """Test a binary annotation."""
        metadata = Metadata(entity=IdentifierEntityParser(key="pubmed", prefix="pubmed"))
        document = dedent(f"""\
            pubmed	{DEFAULT_ANNOTATION_KEY}	curator	date
            30380085	true	orcid:0009-0009-3402-8013	2025-10-07
            30380086	false	orcid:0009-0009-3402-8013	2025-10-07
        """)
        records = self._read(document, metadata=metadata)
        self.assertEqual(2, len(records))

        true_record = records[0]
        self.assertEqual("pubmed:30380085", true_record.reference.curie)
        self.assertEqual({DEFAULT_ANNOTATION_KEY}, set(true_record.annotations))
        annotation = true_record.annotations[DEFAULT_ANNOTATION_KEY]
        self.assertIsInstance(annotation, BinaryAnnotation)
        self.assertTrue(annotation.value)

        false_record = records[1]
        self.assertEqual("pubmed:30380086", false_record.reference.curie)
        self.assertEqual({DEFAULT_ANNOTATION_KEY}, set(false_record.annotations))
        annotation = false_record.annotations[DEFAULT_ANNOTATION_KEY]
        self.assertIsInstance(annotation, BinaryAnnotation)
        self.assertFalse(annotation.value)

    def test_binary_custom_column(self) -> None:
        """Test a binary call."""
        key = "key"
        metadata = Metadata(categories=[BinaryAnnotationMetadata(key=key)])
        document = dedent(f"""\
            entity	{key}	curator	date
            pubmed:30380085	true	orcid:0009-0009-3402-8013	2025-10-07
        """)
        record = self._read_one(document, metadata=metadata)
        self.assertEqual("pubmed:30380085", record.reference.curie)
        self.assertEqual({key}, set(record.annotations))
        annotation = record.annotations[key]
        self.assertIsInstance(annotation, BinaryAnnotation)
        self.assertTrue(annotation.value)

    def test_categorical(self) -> None:
        """Test a binary annotation."""
        metadata = Metadata(
            categories=[CategoricalAnnotationMetadata(key=DEFAULT_ANNOTATION_KEY, nullable=False)]
        )
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            pubmed:30380085	a	orcid:0009-0009-3402-8013	2025-10-07
        """)
        record = self._read_one(document, metadata=metadata)
        self.assertEqual("pubmed:30380085", record.reference.curie)
        self.assertEqual({DEFAULT_ANNOTATION_KEY}, set(record.annotations))
        annotation = record.annotations[DEFAULT_ANNOTATION_KEY]
        self.assertIsInstance(annotation, CategoricalAnnotation)
        self.assertEqual("a", annotation.value)

    def test_categorical_invalid(self) -> None:
        """Test a binary annotation."""
        metadata = Metadata(
            categories=[
                CategoricalAnnotationMetadata(
                    key=DEFAULT_ANNOTATION_KEY, nullable=False, allowed=["a", "b", "c"]
                )
            ]
        )
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            pubmed:30380085	d	orcid:0009-0009-3402-8013	2025-10-07
        """)
        with self.assertRaises(ValueError):
            self._read(document, metadata=metadata)

    def test_multiple(self) -> None:
        """Test a binary annotation."""
        metadata = Metadata(
            categories=[
                BinaryAnnotationMetadata(key="binary_call", nullable=False),
                CategoricalAnnotationMetadata(key="categorical_call", nullable=False),
            ]
        )
        document = dedent("""\
            entity	binary_call	categorical_call	curator	date
            pubmed:30380085	true	a	orcid:0009-0009-3402-8013	2025-10-07
        """)
        record = self._read_one(document, metadata=metadata)
        self.assertEqual("pubmed:30380085", record.reference.curie)
        self.assertEqual({"binary_call", "categorical_call"}, set(record.annotations))

        binary_annotation = record.annotations["binary_call"]
        self.assertIsInstance(binary_annotation, BinaryAnnotation)
        self.assertTrue(binary_annotation.value)

        categorical_annotation = record.annotations["categorical_call"]
        self.assertIsInstance(categorical_annotation, CategoricalAnnotation)
        self.assertEqual("a", categorical_annotation.value)

    def test_missing_required_feature_column(self) -> None:
        """Test a required feature whose column is fully missing."""
        metadata = Metadata(features=[FeatureColumn(key="abstract", required=True)])
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            pubmed:30380085	true	orcid:0009-0009-3402-8013	2025-10-07
        """)
        with self.assertRaises(MissingColumnError):
            self._read(document, metadata=metadata)

    def test_missing_required_feature_row(self) -> None:
        """Test a required feature with a missing row."""
        metadata = Metadata(features=[FeatureColumn(key="abstract", required=True)])
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	abstract	curator	date
            pubmed:30380085	true		orcid:0009-0009-3402-8013	2025-10-07
        """)
        with self.assertRaises(ValueError):
            self._read(document, metadata=metadata)

    def test_feature(self) -> None:
        """Test an optional feature."""
        metadata = Metadata(features=[FeatureColumn(key="abstract")])
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date	abstract
            pubmed:30380085	true	orcid:0009-0009-3402-8013	2025-10-07	some text for an abstract
        """)  # noqa:E501
        record = self._read_one(document, metadata=metadata)
        self.assertEqual({"abstract": "some text for an abstract"}, record.features)

    def test_feature_explicit_datatype(self) -> None:
        """Test an optional feature with an explicit datatype."""
        metadata = Metadata(features=[FeatureColumn(key="abstract", datatype=xsd_string)])
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date	abstract
            pubmed:30380085	true	orcid:0009-0009-3402-8013	2025-10-07	some text for an abstract
        """)  # noqa:E501
        record = self._read_one(document, metadata=metadata)
        self.assertEqual({"abstract": "some text for an abstract"}, record.features)

    def test_feature_explicit_datatype_int(self) -> None:
        """Test an optional feature with an explicit datatype."""
        metadata = Metadata(features=[FeatureColumn(key="count", datatype=xsd_integer)])
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date	count
            pubmed:30380085	true	orcid:0009-0009-3402-8013	2025-10-07	5
        """)
        record = self._read_one(document, metadata=metadata)
        self.assertEqual({"count": 5}, record.features)

    def test_missing_optional_feature_row(self) -> None:
        """Test an optional feature."""
        metadata = Metadata(features=[FeatureColumn(key="abstract")])
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date	abstract
            pubmed:30380085	true	orcid:0009-0009-3402-8013	2025-10-07
        """)
        record = self._read_one(document, metadata=metadata)
        self.assertEqual({}, record.features)
