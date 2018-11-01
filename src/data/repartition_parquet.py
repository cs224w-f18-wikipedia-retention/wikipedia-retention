import os
import re
import sys
import logging

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

import click


def get_spark():
    return SparkSession.builder.getOrCreate()


def extract(spark, path):
    return spark.read.parquet(path)


def transform(dataframe, n_partitions):
    return dataframe.orderBy("timestamp").coalesce(n_partitions)


def load(dataframe, path):
    logging.info("writing to {}".format(path))
    dataframe.write.parquet(path, mode="overwrite")


@click.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True),
    default="data/interim/enwiki-meta-parquet",
)
@click.option(
    "--output-path", type=click.Path(), default="data/processed/enwiki-meta-parquet"
)
@click.option("--n-partitions", type=int, default=48)
@click.option("--dry-run/--no-dry-run", default=True)
def main(input_path, output_path, n_partitions, dry_run):
    spark = get_spark()

    edits_rdd = extract(spark, input_path)
    edits_df = transform(edits_rdd, n_partitions)

    if dry_run:
        logging.info("dry run")
        return

    load(edits_df, output_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
