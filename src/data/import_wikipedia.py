import os
import re
import sys
import logging

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

import click


wikipedia_field_names = [
    "revision",
    "category",
    "image",
    "main",
    "talk",
    "user",
    "user_talk",
    "other",
    "external",
    "template",
    "comment",
    "minor",
    "textdata",
]

wikipedia_schema = T.StructType(
    [
        T.StructField(name, T.StringType(), nullable=True)
        for name in wikipedia_field_names
    ]
)


def get_spark():
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    conf = sc._jsc.hadoopConfiguration()
    conf.set("textinputformat.record.delimiter", "\n\n")
    return spark


def _default_value(item):
    return item[1] if len(item) > 1 else None


def process_edit(edit):
    """Process each line in the edit history"""
    entries = [entry.split(" ", 1) for entry in edit.split("\n")]
    values = tuple(map(_default_value, entries))
    return values


def extract(spark, path):
    num_partitions = spark.sparkContext.defaultParallelism * 4
    records = spark.sparkContext.textFile(path, num_partitions).map(process_edit)
    return spark.createDataFrame(records, schema=wikipedia_schema)


def transform(dataframe, limit=None):
    # prepare the revision column
    revision_columns = [
        "article_id",
        "rev_id",
        "article_title",
        "timestamp",
        "username",
        "user_id",
    ]
    revision_query = [
        F.col("_revision").getItem(i).alias(name)
        for i, name in enumerate(revision_columns)
    ]
    dataframe = (
        dataframe.withColumn("_revision", F.split("revision", " "))
        .select(revision_query + dataframe.columns[1:])
        .drop("_revision")
    )

    # add in the proper typing
    typemap = dict(
        [
            ("article_id", "int"),
            ("rev_id", "int"),
            ("timestamp", "timestamp"),
            ("minor", "boolean"),
            ("textdata", "int"),
            ("user_id", "int"),
        ]
    )

    def cast(name):
        if name in typemap:
            return F.col(name).cast(typemap[name])
        return F.col(name)

    cast_query = map(cast, dataframe.columns)
    dataframe = dataframe.select(cast_query)

    # include limits
    if limit:
        dataframe = dataframe.limit(limit)

    return dataframe.withColumn("year", F.year("timestamp")).withColumn(
        "quarter", F.quarter("timestamp")
    )


def load(dataframe, path):
    logging.info("writing to {}".format(path))
    dataframe.write.partitionBy("year", "quarter").parquet(path, mode="overwrite")


@click.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True),
    default="data/raw/enwiki-20080103.main.bz2",
)
@click.option(
    "--output-path", type=click.Path(), default="data/interim/enwiki-meta-parquet"
)
@click.option("--limit", type=int, default=None)
@click.option("--dry-run/--no-dry-run", default=True)
def main(input_path, output_path, limit, dry_run):
    spark = get_spark()

    edits_rdd = extract(spark, input_path)
    edits_df = transform(edits_rdd, limit)

    if dry_run:
        logging.info("dry run")
        return

    load(edits_df, output_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
