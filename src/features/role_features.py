import click
from subprocess import call
from sys import stdout
from pathlib import Path
import tempfile
import shutil
import os

project_dir = Path(__file__).resolve().parents[2]


# git submodule init
# git submodule update
# cd dependencies/snap/examples/rolx && make

@click.command()
@click.option("--rolx-bin", default=str(project_dir/'dependencies/snap/examples/rolx/testrolx'), type=click.Path(exists=True))
@click.option("--input", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option("--lower-bound", type=int, default=2)
@click.option("--upper-bound", type=int, default=3)
def main(rolx_bin, input, output, lower_bound, upper_bound):
    folder = tempfile.mkdtemp()
    call([
        rolx_bin,
        "-i:" + str(Path(input).resolve()),
        "-l:" + str(lower_bound),
        "-u:" + str(upper_bound),
    ], stdout=stdout, cwd=folder)

    shutil.copyfile(folder + "/roles.txt", output)


if __name__ == '__main__':
    main()
