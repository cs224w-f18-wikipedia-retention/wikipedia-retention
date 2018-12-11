from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
input_path = "src/data/processed/aa_full.csv"
assignments = spark.read.option("delimiter","\t").csv(input_path)
spark.registerTempTable(assignments)
nodes = spark.sql("""
SELECT node, rank() OVER (ORDER BY node) AS rank
FROM (
SELECT DISTINCT node
FROM (
SELECT DISTINCT _c0 AS node
FROM assignments
UNION
SELECT DISTINCT _c1 AS node
FROM assignments
) AS A
) AS A
""")
nodes.registerTempTable('nodes')

remap = spark.sql("""
SELECT e1, B.rank AS e2, w
FROM
(
SELECT B.rank AS e1, A._c1 AS e2, A._c2 AS w
FROM assignments A, nodes B
WHERE A._c0 = B.node
) AS A, nodes B
WHERE A.e2 = B.node
""")
remap.registerTempTable('remap')

node_path = 'src/data/processed/article_assignments'
remap_path = 'src/data/processed/article_remap'
nodes.write.option("sep","\t").csv(node_path)
remap.write.option("sep","\t").csv(remap_path)
