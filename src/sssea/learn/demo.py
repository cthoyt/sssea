"""Demo training on catalaix data."""

import click
import pystow
from curies import NamableReference

from sssea.api import Features, get_binary_summary_rows, read
from sssea.learn import predict_binary, train_binary
from sssea.resources import EXAMPLE_METADATA, EXAMPLE_PATH


def _featurize_abstract(reference: NamableReference, features: Features) -> str | None:
    abstract = features.get("abstract")
    if reference.name is None and abstract is None:
        return None
    text = " ".join(x for x in (reference.name, abstract) if x is not None)
    return text


@click.command()
def main() -> None:
    """Demonstrate training a binary classifier."""
    import pandas as pd
    from tabulate import tabulate

    records = read(EXAMPLE_PATH, metadata=EXAMPLE_METADATA)

    click.echo(
        tabulate(
            get_binary_summary_rows(records, metadata=EXAMPLE_METADATA),
            headers=["Annotation", "Positive", "Negative", "Missing"],
            tablefmt="github",
        )
    )

    embedder = pystow.get_sentence_transformer()

    df_to_predict_on = pd.read_csv(
        "/Users/cthoyt/dev/catalaix/app/prioritization/collated.tsv", sep="\t", dtype=str
    )
    df_to_predict_on = df_to_predict_on[["curie", "title"]]
    title_embeddings = embedder.encode(df_to_predict_on["title"], convert_to_numpy=True)

    predict_category = "PEF depolymerization"

    _classifiers, results = train_binary(
        records, key=predict_category, embedder=embedder, featurizer=_featurize_abstract
    )
    click.echo(
        "\n"
        + tabulate(
            [(n, v) for n, _, v in results],
            headers=["ROC-AUC", predict_category],
            tablefmt="github",
        )
    )
    df_to_predict_on[predict_category] = predict_binary(
        _classifiers.logistic_regression, title_embeddings
    )
    df_to_predict_on = df_to_predict_on.sort_values(predict_category, ascending=False)
    click.echo("\n" + df_to_predict_on.head(20).to_markdown(index=False))


if __name__ == "__main__":
    main()
