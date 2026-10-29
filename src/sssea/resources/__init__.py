"""Resources."""

from pathlib import Path

from curies import NamableReference, Prefix

from ..api import BooleanColumn, IdentifierColumn, Metadata

__all__ = [
    "EXAMPLE_METADATA",
    "EXAMPLE_PATH",
]

HERE = Path(__file__).parent.resolve()
EXAMPLE_PATH = HERE.joinpath("papers.tsv")
EXAMPLE_METADATA = Metadata(
    title="Catalaix PET/PEF Papers",
    description="Curation of papers about the polymerization and depolymerization of PET and PEF",
    license=NamableReference(prefix="spdx", identifier="CC0-1.0"),
    entity=IdentifierColumn(key="doi", prefix=Prefix("doi")),
    annotations=[
        BooleanColumn(key="enzyme paper"),
        BooleanColumn(key="PET depolymerization"),
        BooleanColumn(key="PEF depolymerization"),
        BooleanColumn(key="PEF polymerization"),
        BooleanColumn(key="PET polymerization"),
    ],
)
