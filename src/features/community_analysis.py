from pyspark.sql import SparkSession, functions as F
ss = SparkSession.builder.getOrCreate()
proj_parquet = 'user-network-v3.snappy.parquet'
proj_file = 'uuel.tsv'
community_file = 'src/gen-louvain/assign.txt'
#community_file = 'community_assignments_indetermination.csv'
df = ss.read.parquet(proj_parquet)
df.registerTempTable("uu_edges")

dup_edits = ss.sql("SELECT e1,e2 FROM uu_edges UNION SELECT e2,e1 FROM uu_edges")
dup_edits.registerTempTable("dup_edits")

communities = ss.read.format("com.databricks.spark.csv").option("delimiter"," ").load(community_file)
communities.registerTempTable("communities")

csizes = ss.sql("""
SELECT _c1 AS community,
COUNT(*) AS edges,
COUNT(DISTINCT e1) AS size
FROM communities C, dup_edits E
WHERE C._c0 = E.e1
GROUP BY C._c1
ORDER BY size DESC
""")
csizes.registerTempTable("csizes")

cdensity = ss.sql("""
SELECT community, edges, size,
edges/GREATEST(size*(size-1),1) AS density
FROM csizes
ORDER BY size DESC
""")
cdensity.write.option("delimiter",",").format("com.databricks.spark.csv").save("community_stats")
