# script for generating article-article projection
# copied from anthony's notebook mostly, with modifications to split articles by timestamp
from snap_import_user_projection import UnimodalUserProjection # for some reason this fixes spark-submit
from pyspark.sql import SparkSession, functions as F, types as T

spark = SparkSession.builder.getOrCreate()
#spark.addPyFile('src/data/snap_import_user_projection')

input_path = "src/data/processed/enwiki-meta-compact"
# use this as a smaller sample to test
#input_path = "src/data/processed/enwiki-meta-compact/part-00000-b9d9476b-cc88-44c4-8b82-f39efb715f54-c000.snappy.parquet"
model = UnimodalUserProjection(spark).extract(input_path).transform()

block_list = spark.sql("""
with block_list as (
    select
        user_id,
        concat(year(edit_date), '-', quarter(edit_date)) as edit_date,
        collect_set(cast(concat(quarter(edit_date), mod(year(edit_date),10), article_id) AS bigint)) as article_set
    from bipartite
    group by 1,2
)
select
    user_id,
    size(article_set) as n_articles,
    article_set
from block_list
""")

block_list.cache()
from scipy.optimize import fsolve
import numpy as np

n = 700000
epsilon = 0.01

# loop over all n using previous value as seed
any_bound = {}
all_bound = {}
p_one = 1
p_all = 1
for k in range(2,n+1):
    func_one = lambda p: ((1-p) ** (k-1)) / epsilon - 1
    func_any = lambda p: (1 - ((1- ((1-p) ** (k-1))) ** k)) / epsilon - 1
    p_one = fsolve(func_one,p_one)[0]
    p_all = fsolve(func_any,p_all)[0]
    any_bound[k] = p_one
    all_bound[k] = p_all

from pyspark.sql import types as T
from itertools import combinations
from random import random

bounds = any_bound

# returns n=ceil(k(k-1)/2 * p) edges
def get_n_edges(article_set):
    k = len(article_set)
    p = bounds[k]
    n = int(np.ceil(k*(k-1)/2 * p))
    edge_set = set()
    edge_list = []
    while len(edge_set) < n:
        edge = np.sort(np.random.choice(k,2,replace=False))
        es = str(edge)
        if es not in edge_set:
            edge_set.add(es)
            edge_list.append([article_set[edge[0]],article_set[edge[1]]])
    return edge_list

@F.udf(T.ArrayType(T.ArrayType(T.LongType())))
def sample_edges(article_set):
    k = len(article_set)
    if k < 2:
        return []
    if k < 10: # short circuit if small k
        p = bounds[k]
        return [c for c in combinations(sorted(article_set), 2) if random() < p]
    edges = get_n_edges(article_set)
    return edges

edgelist = (
    block_list
    .select(F.explode(sample_edges("article_set")).alias("edges"))
    .select(F.col("edges").getItem(0).alias("e1"), F.col("edges").getItem(1).alias("e2"))
    .groupby("e1", "e2")
    .agg(F.expr("count(*) as weight"))
)

edgelist.write.option("sep","\t").csv("src/data/processed/article-network-full")
