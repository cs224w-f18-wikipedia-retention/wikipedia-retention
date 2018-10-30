from pyspark.sql import functions as F, types as T


def extract(spark, path):
    header = ["article_id", "user_id", "minor", "word_count", "timestamp"]
    schema = T.StructType(
        [
            T.StructField("article_id", T.IntegerType(), True),
            T.StructField("user_id", T.StringType(), True),
            T.StructField("minor", T.IntegerType(), True),
            T.StructField("word_count", T.IntegerType(), True),
            T.StructField("timestamp", T.TimestampType(), True),
        ]
    )
    return spark.read.csv(path, schema=schema, sep="\t")


def transform(dataframe):
    # Add date partitions and cast user_ids
    cleaned = (
        dataframe.withColumn("year", F.year("timestamp"))
        .withColumn("quarter", F.quarter("timestamp"))
        .withColumn("user_id", F.col("user_id").cast("int"))
        .orderBy("timestamp")
    )
    return cleaned


def load(dataframe, path):
    # this is pretty quick
    (
        dataframe.repartition("year", "quarter")
        .write.partitionBy("year", "quarter")
        .parquet(path, mode="overwrite")
    )


if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    input_path = "../../data/processed/enwiki-20080103"
    output_path = "../../data/processed/enwiki-cleaned"

    df = extract(spark, input_path)
    df = transform(df)
    load(df, output_path)
