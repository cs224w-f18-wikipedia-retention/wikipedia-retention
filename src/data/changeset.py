"""Reference Differences

This script calculates the difference between references of successive edits.
Each revision in the enwiki-meta contains a full set of references from several types of wiki pages.
The article citation network is created by an addition and deletion of references to other articles in main.

These methods generate the proper diffs to include with the revision history.
"""

from pyspark.sql import Window
from pyspark.sql.functions import (
    lag, array_except, array_intersect, array_contains, size, when, explode
)


def transform_changeset(enwiki_meta):
    """Generate a changeset within article_id, rev_id order"""

    previous_revision = Window.partitionBy("article_id").orderBy("rev_id")
    changeset = (
        enwiki_meta
        .withColumn("prev_main", lag("main").over(previous_revision))
        .withColumn("changeset_size", size("main") - size("prev_main"))
        .withColumn("changeset_add", array_except("main", "prev_main"))
        .withColumn("changeset_remove", array_except("prev_main", "main"))
        .drop("prev_main")
    )
    return changeset


def generate_changeset_span(changeset_df):
    """Generate the revision of the next edit removing a reference.

    # TODO: support window sizes that aren't unbounded preceeded (some period T)
    """

    raise NotImplementedError()

    past_revision = (
        Window
        .partitionBy("article_id")
        .orderBy("rev_id")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    edges = (
        changeset_df
        .where("changeset_size <> 0")
        .select("article_id", "rev_id", "changeset_add", "changeset_remove")
        .withColumn("past_rev_id", lag("rev_id").over(past_revision))
        .withColumn("past_changeset_add", lag("changeset_add").over(past_revision))
        .withColumn("removed", explode("changeset_remove"))
        .where("removed isin past_changeset_add")
    )
