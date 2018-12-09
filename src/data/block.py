# script for generating user-user projection.
# copied from anthony's notebook mostly, with modifications to split users by timestamp
from src.data.snap_import_user_projection import UnimodalUserProjection
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.getOrCreate()

input_path = "src/data/processed/enwiki-meta-compact"
# use this as a smaller sample to test
#input_path = "src/data/processed/enwiki-meta-compact/part-00000-b9d9476b-cc88-44c4-8b82-f39efb715f54-c000.snappy.parquet"
model = UnimodalUserProjection(spark).extract(input_path).transform()

# modify to append quarter and last digit of year to user id
# note that conversion process is quite slow if full year is included (100x more nodes to not relabel)
block_list = spark.sql("""
with block_list as (
    select
        article_id,
        concat(year(edit_date), '-', quarter(edit_date)) as edit_date,
        collect_set(cast(concat(quarter(edit_date), mod(year(edit_date),10), user_id) AS bigint)) as user_set
    from bipartite
    group by 1,2
)
select
    article_id,
    edit_date,
    size(user_set) as n_users,
    user_set
from block_list
""")

block_list.cache()

# calculate markov bounds for cliques of size 1-n based on variables
from scipy.optimize import fsolve
import numpy as np
from pyspark.sql import types as T
from itertools import combinations
from random import random

n = 1732
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

@F.udf(T.ArrayType(T.ArrayType(T.LongType())))
def all_edges(user_set):
    return list(combinations(sorted(user_set), 2))

#block_list.selectExpr("n_users*(n_users-1)/2 as n_edges").selectExpr("sum(n_edges)").show()

bounds = any_bound

# returns n=ceil(k(k-1)/2 * p) edges
def get_n_edges(user_set, k, p):
    edge_set = set()
    edge_list = []
    n = int(np.ceil(k*(k-1)/2 * p))
    while len(edge_set) < n:
        edge = np.sort(np.random.choice(k,2,replace=False))
        es = str(edge)
        if es not in edge_set:
            edge_set.add(es)
            edge_list.append(edge)
    return np.array(edge_list)

@F.udf(T.ArrayType(T.ArrayType(T.LongType())))
def sample_edges(user_set):
    k = len(user_set)
    if k < 2:
        return []
    p = bounds[k]
    return [c for c in combinations(sorted(user_set), 2) if random() < p]

edgelist = (
    block_list
    .select(F.explode(sample_edges("user_set")).alias("edges"))
    .select(F.col("edges").getItem(0).alias("e1"), F.col("edges").getItem(1).alias("e2"))
    .groupby("e1", "e2")
    .agg(F.expr("count(*) as weight"))
)
edgelist.write.option("sep"," ").csv("src/data/processed/user-network-full")


