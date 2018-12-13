from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
ind_file = 'src/gen-louvain/user_ind_assign.txt'
remap_file = 'src/gen-louvain/graph_labels.txt'
assign_df = spark.read.option("delimiter"," ").csv(ind_file)
remap_df = spark.read.option("delimiter"," ").csv(remap_file)
assign_df.registerTempTable("assigns")
remap_df.registerTempTable("remaps")

map_df = spark.sql("""
SELECT user,
CAST(LEFT(user, 1) AS integer) AS quarter,
CAST(concat('200',RIGHT(LEFT(user, 2),1)) AS integer) AS year,
CAST(RIGHT(user, LENGTH(user) - 2) AS integer) AS user_id, community_id
FROM (
SELECT CAST(A._c0 AS string) AS user, B._c1 AS community_id
FROM remaps A, assigns B
WHERE A._c1 = B._c0
) AS A
""")

output_path = "src/data/processed/user_communities"
map_df.write.option("sep",",").csv(output_path)
