import os
import re
import sys
from pyspark.sql import SparkSession


def process_edit(edit):
    lines = edit.split('\n')
    header_line = lines[0]
    _revision, article_id, rev_id, article_title, timestamp, _username, user_id = header_line.split(' ')
    minor = parse_value_line(lines[11])
    word_count = parse_value_line(lines[12])
    processed_arr = [article_id,user_id,minor,word_count,timestamp]
    return ' '.join(map(str,processed_arr))

# Given line "NAME VALUE"
# Gets VALUE
# returns '' if VALUE missing
def parse_value_line(line):
    arr = line.split(' ')
    if len(arr) > 1:
        return arr[1]
    else:
        return ''


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
