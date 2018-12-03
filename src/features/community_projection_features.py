from pyspark.sql import SparkSession, functions as F
ss = SparkSession.builder.getOrCreate()
proj_parquet = 'user-network-v3.snappy.parquet'
community_file = 'community_data/community_assignments_indetermination.csv'
user_file = 'base_features_reg.csv'
# load and duplicate edge list
df = ss.read.parquet(proj_parquet)
df.registerTempTable("edges")
dup_edges = ss.sql("SELECT e1,e2 FROM edges UNION SELECT e2,e1 FROM edges")
dup_edges.registerTempTable("dup_edges")
# load community and user features and name columns
communities = ss.read.format("com.databricks.spark.csv").option("delimiter"," ").load(community_file)
communities.registerTempTable("communities")
user_features = ss.read.format("com.databricks.spark.csv").option("delimiter",",").load(user_file)
user_features.registerTempTable("user_features")
# join comms/users
cu = ss.sql("""
SELECT U._c0 AS u_id, C._c1 AS c_id, U._c20 AS contrib
FROM communities C, user_features U
WHERE U._c0 = C._c0
""")
cu.registerTempTable("cu")
# get basic community features
cf1 = ss.sql("""
SELECT c_id, COUNT(*) AS members,
SUM(contrib) AS sum_contrib,
SUM(LOG(contrib+1)) AS sum_lcontrib
FROM cu
GROUP BY c_id
""")
cf1.registerTempTable("cf1")
# get interactions with other communities
cf2 = ss.sql("""
SELECT A.u_id,
AVG(sum_contrib/members) AS avg_icontrib,
AVG(sum_lcontrib/members) AS avg_licontrib
FROM (
SELECT U.u_id, C.c_id, COUNT(*) AS num
FROM dup_edges E, cu U, cu C
WHERE E.e1 = U.u_id AND E.e2 = C.u_id AND C.c_id != U.c_id
GROUP BY U.u_id, C.c_id
) AS A, cf1 C
WHERE A.c_id = C.c_id
GROUP BY A.u_id
""")
cf2.registerTempTable("cf2")
# join and save
cf3 = ss.sql("""
SELECT U.u_id, avg_icontrib, avg_licontrib,
(sum_contrib - contrib) / GREATEST(members-1,1) AS avg_contrib,
(sum_lcontrib - LOG(contrib+1)) / GREATEST(members-1,1) AS avg_lcontrib
FROM cf1 A, cf2 B, cu U
WHERE A.c_id = U.c_id AND B.u_id = U.u_id
""")
cf3.write.option("delimiter",",").format("com.databricks.spark.csv").save("community_features")
