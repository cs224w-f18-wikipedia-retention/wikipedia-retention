import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

input_file = 'processed-all.tsv'

# format is article_id user_id is_minor word_count timestamp
def parse_line_article(line):
    return line.split(' ')[0]

def parse_line_user(line):
    return line.split(' ')[1]

def parse_line_wordcount(line):
    return line.split(' ')[3]

# (user_id, word_count)
def parse_line_user_words(line):
    line_arr = line.split(' ')
    return (line_arr[1],int(line_arr[3]))

# converts tuple rdd (x,y) to a 2 by n_tuples numpy array
def rdd_to_np(rdd):
    return np.array(list(map(lambda x: list(x), rdd.collect())))

# sorts and plots 2xm array with corresponding x/y entries
def plot_arr(arr,loglog=True):
    arr_x = arr[:,0]
    arr_y = arr[:,1]
    idx = np.argsort(arr_x)
    arr_x = arr_x[idx]
    arr_y = arr_y[idx]
    if loglog:
        arr_x = np.log(arr_x)
        arr_y = np.log(arr_y)
    plt.plot(arr_x,arr_y)
    plt.show()

# takes a generic wordcount/sum-like rdd: (key,number)
# reverses it and aggregates by number
def rev_wc(rdd):
    rev_counts = rdd.map(lambda wc: (wc[1],1))
    rev_dist = rev_counts.reduceByKey(lambda n1, n2: n1+n2)
    return rdd_to_np(rev_dist)

lines = sc.textFile(input_file)
# get user distribution
users = lines.map(lambda line: (parse_line_user(line),1))
user_counts = users.reduceByKey(lambda n1, n2: n1+n2)
ucd = rev_wc(user_counts)

# get article change distribution
articles = lines.map(lambda line: (parse_line_article(line),1))
article_counts = articles.reduceByKey(lambda n1, n2: n1+n2)
acd = rev_wc(article_counts)

# get word count distribution
word_counts = lines.map(lambda line: (parse_line_wordcount(line),1))
word_count_counts = word_counts.reduceByKey(lambda n1, n2: n1+n2)
wcd = rev_wc(word_counts)

# get per-user word count distribution
user_word_counts = lines.map(lambda line: parse_line_user_words(line))
uwcd = rev_wc(user_word_counts)

# plotting, only run outside of main
if __name__ != "__main__":
    plot_arr(ucd)
    plot_arr(acd)
    plot_arr(wcd)
    plot_arr(uwcd)
