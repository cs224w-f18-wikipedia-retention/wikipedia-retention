import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

input_file = 'processed-noip.tsv'

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
def plot_arr(arr,filename,title,xlabel,ylabel,loglog=True):
    arr_x = arr[:,0]
    arr_y = arr[:,1]
    idx = np.argsort(arr_x)
    arr_x = arr_x[idx]
    arr_y = arr_y[idx]
    if loglog:
        arr_x = np.log10(arr_x)
        arr_y = np.log10(arr_y)
    plt.plot(arr_x,arr_y,marker='.', linestyle='None')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

def plot_time_hist(arr):
    arr_x = np.array(list(map(lambda x: int(x[1]),arr)))
    plt.hist(umd_x/3600/24,bins=20)
    plt.yscale('log',nonposy='clip')
    plt.xlabel('Days between first and last contribution')
    plt.ylabel('Bin count')
    plt.savefig('time_hist.png')

# takes a generic wordcount/sum-like rdd: (key,number)
# reverses it and aggregates by number
def rev_wc(rdd):
    rev_counts = rdd.map(lambda wc: (wc[1],1))
    rev_dist = rev_counts.reduceByKey(lambda n1, n2: n1+n2)
    return rdd_to_np(rev_dist)

def convert_time(t_str):
    return int(time.mktime(time.strptime(t_str, '%Y-%m-%dT%H:%M:%SZ')))

def parse_line_user_time(line):
    line_arr = line.split(' ')
    t = convert_time(line_arr[4])
    return (line_arr[1],(t,t))

# min of [0], max of [1]
def minmax(tuple1,tuple2):
    return (min(tuple1[0],tuple2[0]),max(tuple1[1],tuple2[1]))

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
wcd = rev_wc(word_count_counts)

# get per-user word count distribution
user_word_counts = lines.map(lambda line: parse_line_user_words(line))
uwcd = rev_wc(user_word_counts)

# get distribution of time from users first to last contribution (hist)
user_times = lines.map(lambda line: parse_line_user_time(line))
user_minmax_times = user_times.reduceByKey(lambda n1, n2: minmax(n1,n2))
user_minmax_diff = user_minmax_times.map(lambda n: (n[0],int(n[1][1]) - int(n[1][0])))
umd = user_minmax_diff.collect()

# plotting
user_params = [
    ucd,
    'user_dist.png',
    'User Edit Counts',
    'User Edit Count (log10)',
    '# Users (log10)'
]
article_params = [
    acd,
    'article_dist.png',
    'Article Edit Counts',
    'Article Edit Count (log10)',
    '# Articles (log10)'
]
word_params = [
    wcd,
    'edit_dist.png',
    'Edit Word Counts',
    'Edit Word Count (log10)',
    '# Edits (log10)'
]
user_word_params = [
    uwcd,
    'user_word_dist.png',
    'User Total Edit Words',
    'User Total Edit Words (log10)',
    '# Users (log10)'
]
plot_arr(*user_params)
plot_arr(*word_params)
plot_arr(*article_params)
plot_arr(*user_word_params)
plot_time_hist(umd)

