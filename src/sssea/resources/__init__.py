"""Resources."""

from pathlib import Path
from curies import Prefix, NamableReference

from sssea.api import Metadata, IdentifierEntityParser, BinaryAnnotationMetadata

__all__ = [
    "EXAMPLE_PATH",
    "EXAMPLE_METADATA",
]

HERE = Path(__file__).parent.resolve()
EXAMPLE_PATH = HERE.joinpath("papers.tsv")
EXAMPLE_METADATA = Metadata(
    title="Catalaix PET/PEF Papers",
    description="Curation of papers about the polymerization and depolymerization of PET and PEF",
    license=NamableReference(prefix="spdx", identifier="CC0-1.0"),
    entity=IdentifierEntityParser(key="doi", prefix=Prefix("doi")),
    categories=[
        BinaryAnnotationMetadata(key="enzyme paper"),
        BinaryAnnotationMetadata(key="PET depolymerization"),
        BinaryAnnotationMetadata(key="PEF depolymerization"),
        BinaryAnnotationMetadata(key="PEF polymerization"),
        BinaryAnnotationMetadata(key="PET polymerization"),
    ],
)
