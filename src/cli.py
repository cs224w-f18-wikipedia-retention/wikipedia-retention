import click
from src.data import wikipedia, projection

cli = click.CommandCollection(sources=[wikipedia.cli, projection.cli])

if __name__ == "__main__":
    cli()
