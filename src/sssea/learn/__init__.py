"""Learning from SSSEA."""

import logging
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple, cast

import numpy as np
import pystow
from numpy._typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sssea.api import (
    DEFAULT_ANNOTATION_KEY,
    BinaryAnnotation,
    CategoricalAnnotation,
    Featurizer,
    Record,
)
from sssea.learn.binary_classification import (
    BinaryClassifierResult,
    BinaryClassifiers,
    predict_binary,
)

__all__ = [
    "predict_binary",
    "prepare_binary",
    "train_binary",
]

logger = logging.getLogger(__name__)


def train_binary(
    records: Iterable[Record],
    *,
    featurizer: Featurizer,
    category: str | None = None,
    embedder: SentenceTransformer | None = None,
) -> tuple[BinaryClassifiers, list[BinaryClassifierResult]]:
    """Train a binary classifier based on positive and negative identifier sets."""
    x, y = prepare_binary(records, category=category, embedder=embedder, featurizer=featurizer)
    return _train_binary(x, y)


def _train_binary(
    x: NDArray[np.float64], y: NDArray[np.bool], *, test_size: float = 0.33, random_state: int = 42
) -> tuple[BinaryClassifiers, list[BinaryClassifierResult]]:
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    classifiers = BinaryClassifiers.make()
    results = classifiers.fit_evaluate(x_train, y_train, x_test, y_test)
    return classifiers, results


class PreparedData(NamedTuple):
    """A pair of inputs (X) and corresponding labels (y)."""

    x: NDArray[np.float64]
    y: NDArray[np.bool]


def prepare_binary(
    records: Iterable[Record],
    *,
    featurizer: Featurizer,
    category: str | None = None,
    embedder: SentenceTransformer | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
    """Prepare SSSEA records for training a binary classifier."""
    if embedder is None:
        embedder = pystow.get_sentence_transformer()

    positives, negatives = stratify_featurize_binary(
        records, category=category, featurizer=featurizer
    )
    if not positives or not negatives:
        raise ValueError("missing one of positives/negatives. check that category is set properly")

    positive_embeddings, positive_indicator = _embed_text(positives, embedder, True)
    negative_embeddings, negative_indicator = _embed_text(negatives, embedder, False)

    x = np.vstack((positive_embeddings, negative_embeddings))
    y = np.hstack((positive_indicator, negative_indicator))
    return PreparedData(x, y)


def _embed_text(
    texts: Iterable[str],
    embedder: SentenceTransformer | TfidfVectorizer,
    positive: bool,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
    embeddings = _embed_text_helper(embedder, texts, **kwargs)
    indicator = np.array([positive] * embeddings.shape[0], dtype=np.bool)
    return embeddings, indicator


def _embed_text_helper(
    embedder: SentenceTransformer | TfidfVectorizer, texts: Iterable[str], **kwargs: Any
) -> NDArray[np.float64]:
    if isinstance(embedder, SentenceTransformer):
        return cast(
            NDArray[np.float64], embedder.encode(list(texts), convert_to_numpy=True, **kwargs)
        )
    elif isinstance(embedder, TfidfVectorizer):
        if kwargs:
            logger.warning("TF-IDF vectorizer takes no kwargs")
        return cast(NDArray[np.float64], embedder.transform(texts))
    else:
        raise TypeError(f"embedder type {type(embedder)} is not supported")


def stratify_featurize_binary(
    records: Iterable[Record],
    *,
    featurizer: Featurizer,
    category: str | None = None,
) -> tuple[Sequence[str], Sequence[str]]:
    """Stratify records based on a binary category."""
    positive = []
    negative = []
    if category is None:
        category = DEFAULT_ANNOTATION_KEY
    for record in records:
        text = featurizer(record.reference, record.features)
        if text is None:
            continue

        match record.annotations.get(category):
            case None:
                continue
            case BinaryAnnotation() as annotation:
                if annotation.value:
                    positive.append(text)
                else:
                    negative.append(text)
            case CategoricalAnnotation():
                # this could be implemented by passing
                # a specific one of the tags to consider
                # as either positive or negative, then
                # considering it as a one-vs-all scenario
                raise NotImplementedError

    return positive, negative
