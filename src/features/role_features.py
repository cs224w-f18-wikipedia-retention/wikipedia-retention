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


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--rolx-bin",
    default=str(project_dir / "dependencies/snap/examples/rolx/testrolx"),
    type=click.Path(exists=True),
)
@click.option("--input", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(exists=False), required=True)
def role_mine(rolx_bin, input, output):
    folder = tempfile.mkdtemp()
    call(
        [rolx_bin, "-i:" + str(Path(input).resolve())],
        stdout=stdout,
        cwd=folder,
    )

    shutil.copyfile(folder + "/mappings.txt", output + "-mappings")
    shutil.copyfile(folder + "/v.txt", output + "-v")


from src.data.snap_import_user_projection import UnimodalUserProjection
from pyspark.sql import SparkSession, functions as F
import pandas as pd

@cli.command()
@click.option(
    "--enwiki-meta",
    type=click.Path(exists=True),
    default=str(project_dir / "data/processed/enwiki-meta-compact"),
)
@click.option(
    "--rolx-roles",
    type=click.Path(exists=True),
    default=str(project_dir / "data/processed/rolx-roles"),
)
@click.option(
    "--output",
    type=click.Path(exists=False),
    default=str(project_dir/"data/processed/role-features")
)
def average_roles(enwiki_meta, rolx_roles, output):
    spark = SparkSession.builder.getOrCreate()
    model = UnimodalUserProjection(spark).extract(enwiki_meta).transform()

    role_df = spark.read.csv(
        "data/processed/rolx-roles",
        schema="user_id INT, role_id INT",
        sep="\t",
        comment="-",
        ignoreLeadingWhiteSpace=True,
    )

    bipartite = (
        spark.table("bipartite")
        .join(role_df, on="user_id", how="left")
        .na.fill({"role_id": -1})
    )

    article_roles = (
        bipartite.groupby("article_id", "edit_date")
        .pivot("role_id")
        .agg(F.count("user_id").alias("n_users"))
        .fillna(0)
    )

    totals = bipartite.groupby("article_id", "edit_date").agg(
        F.count("user_id").alias("deg")
    )

    normalized = article_roles.join(totals, on=["article_id", "edit_date"]).select(
        "article_id",
        "edit_date",
        *[
            (F.col(x) / F.col("deg")).alias(f"role_{x}")
            for x in article_roles.columns[2:]
        ],
    )

    user_roles = (
        bipartite.join(normalized, on=["article_id", "edit_date"])
        .groupby("user_id")
        .agg(*[F.sum(x).alias(x) for x in normalized.columns[2:]])
    )

    # run row-wise normalization
    avg_roles = user_roles.toPandas()
    df = avg_roles.iloc[:, 1:]
    # https://stackoverflow.com/a/35679163
    df = pd.concat([avg_roles.iloc[:,1], df.div(df.sum(axis=0), axis=1)], axis=1)
    df.to_csv(output)

if __name__ == "__main__":
    cli()
