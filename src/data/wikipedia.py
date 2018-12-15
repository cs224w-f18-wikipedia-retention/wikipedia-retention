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

    cast_query = list(map(cast, dataframe.columns))
    dataframe = dataframe.select(cast_query)

    to_array = {
        "category",
        "image",
        "main",
        "talk",
        "user",
        "user_talk",
        "other",
        "external",
        "template",
    }

    def split(name):
        if name in to_array:
            return F.split(name, " ").alias(name)
        return F.col(name)

    split_query = list(map(split, dataframe.columns))
    dataframe = dataframe.select(split_query)

    # include limits
    if limit:
        dataframe = dataframe.limit(limit)

    return dataframe.withColumn("year", F.year("timestamp")).withColumn(
        "quarter", F.quarter("timestamp")
    )


def load(dataframe, path, mode):
    logging.info("writing to {}".format(path))
    dataframe.write.partitionBy("year", "quarter").parquet(path, mode=mode)


@click.group(name="wikipedia")
def cli():
    pass


@cli.command()
@click.option("--input-path", type=click.Path(exists=True), required=True)
@click.option("--output-path", type=click.Path(), required=True)
@click.option("--limit", type=int, default=None)
@click.option("--overwrite/--no-overwrite", default=False)
def bz2parquet(input_path, output_path, limit, overwrite):
    spark = get_spark()
    edits_rdd = extract(spark, input_path)
    edits_df = transform(edits_rdd, limit)
    load(edits_df, output_path, "overwrite" if overwrite else "error")


@cli.command()
@click.option("--input-path", type=click.Path(exists=True), required=True)
@click.option("--output-path", type=click.Path(), required=True)
@click.option("--n-partitions", type=int, default=48)
@click.option("--overwrite/--no-overwrite", default=False)
def coalesce(input_path, output_path, n_partitions, overwrite):
    spark = get_spark()

    mode = "overwrite" if overwrite else "error"
    (
        spark.read.parquet(input_path)
        .orderBy("year", "quarter", "article_id", "timestamp")
        .coalesce(n_partitions)
        .write.parquet(output_path, mode=mode)
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
