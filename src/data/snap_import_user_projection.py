import glob
import os
import re
import shutil
import sys
import logging

from pyspark.sql import SparkSession

import click


class UnimodalUserProjection:
    def __init__(self, spark=None):
        self.spark = spark or SparkSession.builder.getOrCreate()

    def extract(self, path):
        df = self.spark.read.parquet(path)
        df.createOrReplaceTempView("enwiki")
        return self

    def transform(self, period):
        self.create_bipartite_edgelist(period)
        self.unimodal_user_projection()
        # registers table `projection`
        return self

    def load(self, name):
        dataframe = self.spark.table("projection")

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
        query ="""
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
            count(*) as num_edits,
            edit_date
        FROM
            subset
        WHERE
            subset.user_id is not null
        GROUP BY 1, 2, 3"""
        edges = self.spark.sql(query)
        if period:
            year, quarter = period.split('-')
            edges = (
                edges
                .where("year(edit_date) = {}".format(year))
                .where("quarter(edit_date) = {}".format(quarter))
            )
        edges.createOrReplaceTempView("bipartite")

    def unimodal_user_projection(self):
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
        projection.createOrReplaceTempView("projection")


@click.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True),
    default="data/processed/enwiki-meta-compact",
)
@click.option("--output-suffix", type=click.Path(), default="enwiki-projection-user")
@click.option("--period", type=str)
def main(input_path, output_suffix, period):
    transformer = UnimodalUserProjection().extract(input_path).transform(period)
    if period:
        name = "{}-{}".format(period, output_suffix)
    else:
        name = output_suffix
    transformer.load(name)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
