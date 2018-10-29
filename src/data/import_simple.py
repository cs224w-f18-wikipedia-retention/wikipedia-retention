import re
import sys
from pyspark import SparkConf, SparkContext
import os

conf = SparkConf()
sc = SparkContext(conf=conf)
conf = sc._jsc.hadoopConfiguration()
conf.set("textinputformat.record.delimiter", "\n\n")

compressed_file = 'enwiki-20080103.main.bz2'
output_file = 'wiki-processed-noip/'
edits_per_round = 1000000
lines_per_edit = 14
rounds = 0

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

parsed_edits = sc.textFile(compressed_file)
processed_edits = parsed_edits.map(lambda edit: process_edit(edit), parsed_edits)
# remove users with id starting with ip: (signifying anon)
process_edits_noip = parsed_edits.filter(lambda row: row[1][0] != 'i')
processed_edits.saveAsTextFile(output_file)
