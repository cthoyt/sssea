"""Demo training on catalaix data."""

import click
import pystow
from curies import NamableReference, Prefix

from sssea.api import BinaryAnnotationMetadata, Features, IdentifierEntityParser, Metadata, read
from sssea.learn import predict_binary, summarize, train_binary


@click.command()
def main() -> None:
    """Demonstrate training a binary classifier."""
    from tabulate import tabulate

    metadata = Metadata(
        title="test",
        description="test",
        entity=IdentifierEntityParser(key="doi", prefix=Prefix("doi")),
        categories=[
            BinaryAnnotationMetadata(key="enzyme paper"),
            BinaryAnnotationMetadata(key="PET depolymerization"),
            BinaryAnnotationMetadata(key="PEF depolymerization"),
            BinaryAnnotationMetadata(key="PEF polymerization"),
            BinaryAnnotationMetadata(key="PET polymerization"),
        ],
    )

    records = read("/Users/cthoyt/dev/catalaix/app/sources/citavi/papers.tsv", metadata=metadata)
    click.echo(summarize(records, metadata=metadata))
    embedder = pystow.get_sentence_transformer()
    _classifiers, results = train_binary(
        records, category="PEF depolymerization", embedder=embedder, featurizer=_featurize_abstract
    )
    click.echo(
        "\n"
        + tabulate([(n, v) for n, _, v in results], headers=["model", "ROC-AUC"], tablefmt="github")
    )

    import pandas as pd

    df = pd.read_csv(
        "/Users/cthoyt/dev/catalaix/app/prioritization/collated.tsv", sep="\t", dtype=str
    )

    vecs = embedder.encode(df["title"], convert_to_numpy=True)
    df["PEF depolymerization"] = predict_binary(_classifiers.logistic_regression, vecs)
    df = df.sort_values("PEF depolymerization", ascending=False)
    click.echo("\n" + df.head(20).to_markdown(index=False))
    df.to_csv("pef-depolymerization.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()


def _featurize_abstract(reference: NamableReference, features: Features) -> str | None:
    abstract = features.get("abstract")
    if reference.name is None and abstract is None:
        return None
    text = " ".join(x for x in (reference.name, abstract) if x is not None)
    return text
