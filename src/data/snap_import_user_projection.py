import glob
import os
import re
import shutil
import sys
import logging
from functools import partial
from itertools import combinations

from pyspark.sql import SparkSession, functions as F, types as T
from scipy.optimize import fsolve
import numpy as np

import click


def markov_bound(n, epsilon, method="any"):
    # see src/data/gen_markov_bounds
    # loop over all n using previous value as seed
    if method == "any":
        func = lambda k, p: ((1 - p) ** (k - 1)) / epsilon - 1
    elif method == "all":
        func = lambda k, p: (1 - ((1 - ((1 - p) ** (k - 1))) ** k)) / epsilon - 1
    else:
        raise ValueError("invalid method: {}; must be one of (any, all)".format(method))

    bound = {}
    p = 1
    for k in range(2, n + 1):
        p = fsolve(partial(func, k), p)[0]
        bound[k] = p
    return bound


class UnimodalUserProjection:
    def __init__(self, spark=None):
        self.spark = spark or SparkSession.builder.getOrCreate()

    def extract(self, path):
        df = self.spark.read.parquet(path)
        df.createOrReplaceTempView("enwiki")
        return self

    def transform(self, period=None, epsilon=0.01):
        self.create_bipartite_edgelist(period)

        # register views
        self.daily_block_projection()
        self.reduced_quarterly_block_projection(epsilon)
        return self

    def load(self, view, name):
        dataframe = self.spark.table(view)

        interim_path = "data/interim/{}".format(name)
        logging.info("writing to {}".format(interim_path))
        dataframe.coalesce(1).write.csv(interim_path, sep="\t", mode="overwrite")

        # move the file to the final location
        processed_file = "data/processed/{}.csv".format(name)
        logging.info("moving file to processed/{}")
        interim_file = glob.glob("{}/*.csv".format(interim_path))[0]
        shutil.copy(interim_file, processed_file)

        # cleanup
        logging.info("cleaning up: {}".format(interim_path))
        shutil.rmtree(interim_path)

    def create_bipartite_edgelist(self, period=None):
        """Create a weighted user-article graph"""
        query = """
        with subset as (
            SELECT
                date_format(timestamp, 'yyyy-MM-dd') as edit_date,
                article_id,
                cast(user_id as int) as user_id,
                textdata
            FROM
                enwiki
        )
        -- collect the weighted edge-list
        SELECT
            user_id,
            article_id,
            edit_date,
            sum(log(textdata)) as word_count,
            count(*) as num_edits
        FROM
            subset
        WHERE
            user_id is not null
        GROUP BY 1, 2, 3"""
        edges = self.spark.sql(query)
        if period:
            year, quarter = period.split("-")
            edges = edges.where("year(edit_date) = {}".format(year)).where(
                "quarter(edit_date) = {}".format(quarter)
            )
        edges.createOrReplaceTempView("bipartite")

    def daily_block_projection(self):
        query = """
        WITH unimodal_projection as (
            SELECT
                t1.user_id as e1,
                t2.user_id as e2,
                count(*) as shared_articles
            FROM bipartite t1
            JOIN bipartite t2
            ON t1.article_id = t2.article_id AND t1.edit_date = t2.edit_date
            WHERE t1.user_id < t2.user_id
            GROUP BY 1, 2
        )
        SELECT e1, e2, shared_articles
        FROM unimodal_projection
        """
        projection = self.spark.sql(query)
        projection.createOrReplaceTempView("daily_block_projection")

    def reduced_quarterly_block_projection(self, epsilon):
        block_list = self.spark.sql(
            """
        with block_list as (
            select
                article_id,
                concat(year(edit_date), '-', quarter(edit_date)) as edit_date,
                collect_set(user_id) as user_set
            from bipartite
            group by 1,2
        )
        select
            article_id,
            edit_date,
            size(user_set) as n_users,
            user_set
        from block_list
        """
        )
        block_list.cache()

        # this could also be read from a file, but it seems fine to solve on the spot
        n = block_list.selectExpr("max(n_users) as n").collect()[0].n
        bounds = markov_bound(n, epsilon)

        @F.udf(T.ArrayType(T.ArrayType(T.IntegerType())))
        def sample_edges(user_set):
            k = len(user_set)
            if k < 2:
                return []
            p = bounds[k]
            edges = [c for c in combinations(sorted(user_set), 2) if random() < p]
            return edges

        projection = (
            block_list.select(F.explode(sample_edges("user_set")).alias("edges"))
            .select(
                F.col("edges").getItem(0).alias("e1"),
                F.col("edges").getItem(1).alias("e2"),
            )
            .groupby("e1", "e2")
            .agg(F.expr("count(*) as weight"))
        )
        projection.createOrReplaceTempView("reduced_quarterly_block_projection")
        block_list.unpersist()


@click.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True),
    default="data/processed/enwiki-meta-compact",
)
@click.option("--output-suffix", type=click.Path(), default="user-network-v3")
@click.option("--period", type=str)
@click.option(
    "--epsilon", type=float, default=0.01, help="markov bound on number of nodes"
)
def main(input_path, output_suffix, period, epsilon):
    transformer = (
        UnimodalUserProjection().extract(input_path).transform(period, epsilon)
    )
    if period:
        name = "{}-{}".format(period, output_suffix)
    else:
        name = output_suffix
    transformer.load("reduced_quarterly_block_projection", name)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
