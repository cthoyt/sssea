"""Command line interface for :mod:`sssea`."""

import click

__all__ = [
    "main",
]


@click.command()
def main() -> None:
    """CLI for sssea."""


if __name__ == "__main__":
    main()
