import os
import re
import sys

from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
import pyspark.sql.types as T


wikipedia_schema = T.StructType([
    # REVISION fields
    T.StructField("article_id",     T.IntegerType(),    nullable=False),
    T.StructField("rev_id",         T.IntegerType(),    nullable=False),
    T.StructField("article_title",  T.StringType(),     nullable=False),
    T.StructField("timestamp",      T.StringType(),     nullable=False),
    T.StructField("username",       T.StringType(),     nullable=False),
    T.StructField("user_id",        T.StringType(),     nullable=True),
    # other fields
    T.StructField("category",       T.StringType(),     nullable=True),
    T.StructField("image",          T.StringType(),     nullable=True),
    T.StructField("main",           T.StringType(),     nullable=True),
    T.StructField("talk",           T.StringType(),     nullable=True),
    T.StructField("user",           T.StringType(),     nullable=True),
    T.StructField("user_talk",      T.StringType(),     nullable=True),
    T.StructField("other",          T.StringType(),     nullable=True),
    T.StructField("external",       T.StringType(),     nullable=True),
    T.StructField("template",       T.StringType(),     nullable=True),
    T.StructField("comment",        T.StringType(),     nullable=True),
    T.StructField("minor",          T.BooleanType(),    nullable=False),
    T.StructField("textdata",       T.IntegerType(),    nullable=False),
])

# The list of fields with a method for casting the string into the appropriate type.
_to_cast = [
    ('article_id', int),
    ('rev_id', int),
    ('minor', bool),
    ('textdata', int)
]


def default_value(item):
    return item if len(item) > 1 else item + [None]


def process_edit(edit):
    """Process each line in the edit history"""

    entries = [entry.split(' ', 1) for entry in edit.split('\n')]
    # lower case keys and add default values, ignore the header
    pairs = {key.lower(): value for key, value in map(default_value, entries[1:])}

    # each REVISION has 6 fields
    revision = entries[0][1]
    revision_columns = ["article_id", "rev_id", "article_title", "timestamp", "username", "user_id"]
    # the user_id is missing sometimes, so add a default value of None
    revision_pairs = zip(revision_columns, revision.split() + [None])
    pairs.update(revision_pairs)

    for name, cast in _to_cast:
        pairs[name] = cast(pairs[name])

    return Row(**pairs)


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    conf = sc._jsc.hadoopConfiguration()
    conf.set("textinputformat.record.delimiter", "\n\n")

    input_file = '../data/raw/enwiki-20080103.main.bz2'
    output_file = '../data/processed/enwiki-20080103/'
    edits_per_round = 1000000
    lines_per_edit = 14
    rounds = 0

    parsed_edits = sc.textFile(input_file)
    processed_edits = parsed_edits.map(process_edit)
    processed_edits.saveAsTextFile(output_file)
