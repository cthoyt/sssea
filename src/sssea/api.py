"""An implementation of SSSEA."""

import datetime
import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar

from curies import NamableReference, Prefix, Reference
from curies.vocabulary import Primitive, parse_by_datatype
from pydantic import BaseModel, Field
from pystow.utils import Source, safe_open_dict_reader

__all__ = [
    "DEFAULT_ANNOTATION_KEY",
    "BinaryAnnotation",
    "BinaryAnnotationMetadata",
    "CURIEEntityParser",
    "CategoricalAnnotation",
    "CategoricalAnnotationMetadata",
    "FeatureConf",
    "Features",
    "Featurizer",
    "IdentifierEntityParser",
    "Metadata",
    "MissingColumnError",
    "Record",
    "read",
]

logger = logging.getLogger(__name__)

AnnotationValueType = TypeVar("AnnotationValueType")

AnnotationType: TypeAlias = Literal["binary", "categorical"]

#: A dictionary of features
Features: TypeAlias = dict[str, Any]

#: A function that consumes a features dictionary and produces a feature string
Featurizer: TypeAlias = Callable[[NamableReference, Features], str | None]

DEFAULT_ENTITY_KEY = "entity"


class _EntityParser(BaseModel, ABC):
    type: Literal["curie", "prefix"]
    key: str

    @abstractmethod
    def parse(self, record: dict[str, Any], *, name: str | None = None) -> NamableReference:
        """Parse the entity."""


class CURIEEntityParser(_EntityParser):
    """A configuration for an entity column represented by CURIEs."""

    type: Literal["curie"] = "curie"

    def parse(self, record: dict[str, Any], *, name: str | None = None) -> NamableReference:
        """Parse the entity."""
        return NamableReference.from_curie(record[self.key], name=name)


class IdentifierEntityParser(_EntityParser):
    """A configuration for an entity column represented by local unique identifiers."""

    type: Literal["prefix"] = "prefix"
    prefix: Prefix

    def parse(self, record: dict[str, Any], *, name: str | None = None) -> NamableReference:
        """Parse the entity."""
        return NamableReference(prefix=self.prefix, identifier=record[self.prefix], name=name)


#: A mechanism for extracting the primary entity
EntityParser = Annotated[CURIEEntityParser | IdentifierEntityParser, Field(discriminator="type")]


def _default_entity() -> EntityParser:
    return CURIEEntityParser(key=DEFAULT_ENTITY_KEY)


class _AnnotationMetadata(BaseModel):
    annotation_type: AnnotationType
    key: str
    predicate: Annotated[
        NamableReference | None,
        Field(default=None, description="the predicate connecting the entity to the annotation"),
    ] = None
    nullable: bool = True


class BinaryAnnotationMetadata(_AnnotationMetadata):
    """Information about how a given category should be parsed."""

    annotation_type: Literal["binary"] = "binary"


class CategoricalAnnotationMetadata(_AnnotationMetadata):
    """Information about how a given category should be parsed."""

    annotation_type: Literal["categorical"] = "categorical"
    allowed: list[str] | None = None


AnnotationMetadata = Annotated[
    BinaryAnnotationMetadata | CategoricalAnnotationMetadata, Field(discriminator="annotation_type")
]

DEFAULT_ANNOTATION_KEY = "call"


def _default_category() -> list[AnnotationMetadata]:
    return [BinaryAnnotationMetadata(key=DEFAULT_ANNOTATION_KEY, nullable=False)]


class FeatureConf(BaseModel):
    """A configuration for parsing a feature."""

    key: str
    datatype: Reference | None = None
    required: bool = False

    def parse(self, value: str) -> Primitive:
        """Parse an XSD primitive."""
        if self.datatype is None:
            return value
        return parse_by_datatype(value, self.datatype)


class Metadata(BaseModel):
    """Configuration for parsing a literature prioritization TSV document.

    The refence can be configured either with a column name for a curie,
    or a column name for an identifier
    """

    title: str | None = None
    description: str | None = None
    license: NamableReference | None = None
    creators: list[NamableReference] | None = None

    entity: EntityParser = Field(default_factory=_default_entity)
    categories: list[AnnotationMetadata] = Field(default_factory=_default_category)
    features: list[FeatureConf] | None = None


class _Annotation(BaseModel, Generic[AnnotationValueType]):
    annotation_type: AnnotationType
    value: AnnotationValueType
    curator: NamableReference | None = None
    date: datetime.date | None = None


class BinaryAnnotation(_Annotation[bool]):
    """A binary judgment."""

    annotation_type: Literal["binary"] = "binary"
    value: bool


class CategoricalAnnotation(_Annotation[str]):
    """A categorical judgment."""

    annotation_type: Literal["categorical"] = "categorical"
    value: str


# TODO percentage, 5-star/Likert?

#: An annotation, either binary or categorical
Annotation = Annotated[
    BinaryAnnotation | CategoricalAnnotation, Field(discriminator="annotation_type")
]


class Record(BaseModel):
    """A single curation record."""

    reference: NamableReference
    annotations: dict[str, Annotation]
    features: Features


def read(path: Source, *, metadata: Metadata | None = None) -> list[Record]:
    """Read a SSSLP TSV file."""
    return list(_read_iter(path, metadata=metadata))


def _parse_bool(x: str) -> bool:
    if x.lower() in {"yes", "true", "on", "1"}:
        return True
    elif x.lower() in {"no", "false", "off", "0"}:
        return False
    raise ValueError(f"Invalid boolean value: {x}")


DOI = Prefix("doi")
PUBMED = Prefix("pubmed")


class MissingColumnError(ValueError):
    """A value error raised when columns are missing."""

    def __init__(self, expected: list[str], actual: Sequence[str]) -> None:
        """Initialize the exception."""
        self.columns = expected
        self.actual = actual

    def __str__(self) -> str:
        return f"missing columns for required annotations: {self.columns}, got {self.actual}"


def _read_iter(path: Source, *, metadata: Metadata | None = None) -> Iterable[Record]:  # noqa:C901
    if metadata is None:
        metadata = Metadata()

    # check that metadata doesn't have duplicate keys
    counter = Counter(category.key for category in metadata.categories)
    duplicates = Counter({k: count for k, count in counter.items() if count > 1})
    if duplicates:
        raise ValueError(f"duplicate categories defined in metadata: {duplicates}")

    with safe_open_dict_reader(path) as reader:
        if not reader.fieldnames:  # pragma: no cover
            raise ValueError("empty file")

        if metadata.entity.key not in reader.fieldnames:
            raise ValueError(f"missing entity key: {metadata.entity.key}")

        if missing_columns := sorted(
            category.key
            for category in metadata.categories
            if not category.nullable and category.key not in reader.fieldnames
        ):
            raise MissingColumnError(missing_columns, reader.fieldnames)

        if missing_feature_columns := sorted(
            feature.key
            for feature in metadata.features or []
            if feature.required and feature.key not in reader.fieldnames
        ):
            raise MissingColumnError(missing_feature_columns, reader.fieldnames)

        # If name and title in reader columns, raise
        # TODO check that the reader's headers include the defined calls and features
        for i, record in enumerate(reader, start=2):
            name = record.get("name") or record.get("title")

            try:
                reference = metadata.entity.parse(record, name=name)
            except ValueError:
                logger.warning("[%d] no entity found", i)
                continue

            features: dict[str, Any] = {}
            for feature in metadata.features or []:
                value = record.get(feature.key)
                if not value:  # catch both explicit none and empty string
                    if feature.required:
                        raise ValueError
                    continue
                features[feature.key] = feature.parse(value)

            annotations: dict[str, Annotation] = {}
            for category in metadata.categories:
                value = record.get(category.key)
                if not value:  # catch both explicit none and empty string
                    if category.nullable:
                        continue
                    else:
                        raise ValueError(f"required to have category: {category.key}")

                curator = (
                    NamableReference.from_curie(curator_curie)
                    if (curator_curie := record.get("curator"))
                    else None
                )

                match category:
                    case BinaryAnnotationMetadata():
                        annotations[category.key] = BinaryAnnotation(
                            value=_parse_bool(value), curator=curator, date=record.get("date")
                        )
                    case CategoricalAnnotationMetadata():
                        if category.allowed and value not in category.allowed:
                            raise ValueError
                        annotations[category.key] = CategoricalAnnotation(
                            value=value, curator=curator, date=record.get("date")
                        )

            if not annotations:
                logger.warning("[row %d] no calls found", i)
                continue

            yield Record(reference=reference, annotations=annotations, features=features)
