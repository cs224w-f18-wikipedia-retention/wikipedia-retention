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
        self.create_edges(period)
        self.unimodal_user_projection(threshold=1)
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

    def create_edges(self, period):
        """Create a weighted user-article graph"""

        year, quarter = period.split("-")
        query = """
        with subset as (
            SELECT
                article_id,
                cast(user_id as int) as user_id,
                textdata
            FROM enwiki
            WHERE year = {} AND quarter = {}
        ),

        -- remove all nodes where the degree is < 2
        degree as (
            SELECT
                user_id,
                count(distinct article_id) as degree
            FROM subset
            GROUP BY 1
        )

        -- collect the weighted edge-list
        SELECT
            subset.user_id,
            article_id,
            sum(textdata) as word_count,
            count(*) as num_edits
        FROM subset
        INNER JOIN degree
        ON subset.user_id = degree.user_id
        WHERE
            degree > 1 AND
            subset.user_id IS NOT NULL
        GROUP BY 1, 2
        """.format(
            year, quarter
        )
        edges = self.spark.sql(query)
        edges.createOrReplaceTempView("edges")

    def unimodal_user_projection(self, threshold=1):
        query = """
        -- TODO: jaccard index instead of common neighbors
        with unimodal_projection as (
            SELECT
                t1.user_id as e1,
                t2.user_id as e2,
                count(*) as shared_articles
            FROM edges t1
            JOIN edges t2 ON t1.article_id = t2.article_id
            GROUP BY 1, 2
        )

        SELECT e1, e2
        FROM unimodal_projection
        WHERE shared_articles > {}
        ORDER BY e1
        """.format(
            threshold
        )
        projection = self.spark.sql(query)
        projection.createOrReplaceTempView("projection")


@click.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True),
    default="data/processed/enwiki-meta-compact",
)
@click.option("--output-suffix", type=click.Path(), default="enwiki-projection")
@click.option("--period", type=str, default="2007-1")
@click.option("--dry-run/--no-dry-run", default=True)
def main(input_path, output_suffix, period, dry_run):
    transformer = UnimodalUserProjection().extract(input_path).transform(period)

    name = "{}-{}".format(period, output_suffix)
    if dry_run:
        # TODO: move this section into `load`
        logging.info("Dry run for {}".format(name))
        logging.info("Rerun the command with `--no-dry-run`")
        return

    transformer.load(name)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
