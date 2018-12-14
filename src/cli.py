import click
from src.data import import_wikipedia as impwiki, projection

cli = click.CommandCollection(sources=[impwiki.cli, projection.cli])

if __name__ == "__main__":
    cli()
