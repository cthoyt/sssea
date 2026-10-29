"""Test learning workflow."""

import unittest

import pystow
from curies import NamableReference
from pystow.utils import requires_package

from sssea.api import Features, read
from sssea.learn import train_binary
from sssea.resources import EXAMPLE_METADATA, EXAMPLE_PATH


def _featurize_abstract(reference: NamableReference, features: Features) -> str | None:
    abstract = features.get("abstract")
    if reference.name is None and abstract is None:
        return None
    text = " ".join(x for x in (reference.name, abstract) if x is not None)
    return text


class TestLearn(unittest.TestCase):
    """Test learning workflow."""

    def test_learn_tfidf(self) -> None:
        """Test learning workflow."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        records = read(EXAMPLE_PATH, metadata=EXAMPLE_METADATA)

        embedder = TfidfVectorizer()
        embedder.fit([r.reference.name for r in records if r.reference.name is not None])

        _classifiers, _results = train_binary(
            records,
            key=EXAMPLE_METADATA.categories[0].key,
            embedder=embedder,
            featurizer=_featurize_abstract,
        )

    @requires_package("sentence_transfomers")
    def test_learn_sentence_transformers(self) -> None:
        """Test learning workflow."""
        records = read(EXAMPLE_PATH, metadata=EXAMPLE_METADATA)

        embedder = pystow.get_sentence_transformer()

        _classifiers, _results = train_binary(
            records,
            key=EXAMPLE_METADATA.categories[0].key,
            embedder=embedder,
            featurizer=_featurize_abstract,
        )
