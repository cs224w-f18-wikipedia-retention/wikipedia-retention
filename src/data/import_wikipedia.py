import os
import re
import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T


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
    "textdata"
]

wikipedia_schema = T.StructType([
    T.StructField(name, T.StringType(), nullable=True)
    for name in wikipedia_field_names
])


def get_spark():
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    conf = sc._jsc.hadoopConfiguration()
    conf.set("textinputformat.record.delimiter", "\n\n")
    return spark


def _default_value(item):
    return item[1] if len(item) > 1 else None


def process(edit):
    entries = [entry.split(' ', 1) for entry in edit.split('\n')]
    values = tuple(map(_default_value, entries))
    return value


def process_edit(edit):
    """Process each line in the edit history

    Returns a tuple of (article_id, user_id, is_minor, wordcount, timestamp)
    """
    values = process(edit)


def extract(spark, path):
    records = spark.sparkContext.textFile(path).map(process_edit)
    return spark.createDataFrame(records, schema=wikipedia_schema)


def transform(dataframe, limit=None):
    # prepare the revision column
    revision_columns = [
        "article_id", "rev_id", "article_title",
        "timestamp", "username", "user_id"
    ]
    revision_query = [
        F.col("_revision").getItem(i).alias(name)
        for i, name in enumerate(revision_columns)
    ]
    dataframe = (
        dataframe
        .withColumn("_revision", F.split("revision", ' '))
        .select(revision_query + dataframe.columns[1:])
        .drop("_revision")
    )

    # add in the proper typing
    typemap = dict([
        ('article_id',  'int'),
        ('rev_id',      'int'),
        ('timestamp',   'timestamp'),
        ('minor',       'boolean'),
        ('textdata',    'int'),
    ])
    def cast(name):
        if name in typemap:
            return F.col(name).cast(typemap[name])
        return F.col(name)

    cast_query = map(cast, dataframe.columns)
    dataframe = dataframe.select(cast_query)

    # include limits
    if limit:
        dataframe = dataframe.limit(limit)

    return (
        dataframe
        .withColumn("year", F.year("timestamp"))
        .withColumn("month", F.month("timestamp"))
    )


def write_parquet(dataframe, path):
    dataframe.write.partitionBy("year", "month").write(path)


if __name__ == '__main__':
    spark = get_spark()
    sc = spark.sparkContext

    input_file = '../../data/raw/enwiki-20080103.main.bz2'
    output_file = '../../data/processed/enwiki-20080103/'
    edits_per_round = 1000000
    lines_per_edit = 14
    rounds = 0

    parsed_edits = sc.textFile(input_file)
    processed_edits = parsed_edits.map(process_edit)
    processed_edits.saveAsTextFile(output_file)
