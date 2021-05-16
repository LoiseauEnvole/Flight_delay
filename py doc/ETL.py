# Databricks notebook source
# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# import libraries
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, Imputer
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql import SQLContext
import pyspark.ml.feature as ftr
from pathlib import Path
 
import pandas as pd
import requests
import timezonefinder
import pandas_profiling
from pandas_profiling.utils.cache import cache_file
 
# import libraries
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, Imputer
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql import SQLContext
import pyspark.ml.feature as ftr
from pathlib import Path
 
import requests
import timezonefinder
import pandas_profiling
from pandas_profiling.utils.cache import cache_file
 
# Libraries for modeling
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import PipelineModel
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import PCA
 
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark


# COMMAND ----------

name_dict = {"JFK": "JFK INTERNATIONAL AIRPORT",
             "BOS": "BOSTON",
             "ATL": "ATLANTA HARTSFIELD",
             "DFW": "DALLAS FORT WORTH",
             "ORD": "CHICAGO OHARE INTERNATIONAL AIRPORT",
             "CLT": "CHARLOTTE DOUGLAS AIRPORT",
             "DCA": "WASHINGTON REAGAN NATIONAL AIRPORT",
             "IAH": "HOUSTON INTERCONTINENTAL",
             "SEA": "SEATTLE",
             "LAX": "LOS ANGELES INTERNATIONAL AIRPORT",
             "SFO": "SAN FRANCISCO INTERNATIONAL AIRPORT",
             "DEN": "DENVER INTERNATIONAL AIRPORT",
             "CVG": "CINCINNATI NORTHERN KENTUCKY INTERNATIONAL AIRPORT",
             "BOI": "CALDWELL",
             "PSP": "DESERT RESORTS REGIONAL AIRPORT",
             "FAT": "MAMMOTH LAKES MAMMOTH YOSEMITE"
            }

l = ["JFK","SEA","BOS","ATL","LAX","SFO","DEN","DFW","ORD","CVG","CLT","DCA","IAH", "BOI", "PSP", "FAT"]
 
city_codes = {'DEN': 'Denver', 'CLT': 'Charlotte', 'SEA': 'Seattle', 'DFW': 'Dallas Fort Worth', 'ATL': 'Atlanta', 'LAX': 'Los Angeles', 'BOS': 'Boston', 'JFK': 'New York', 'IAH': 'Houston', 'DCA': 'Washington', 'ORD': 'Chicago', 'SFO': 'San Francisco', 'CVG': 'Cincinnati'}
 
down_col = ["STATION", "DATE", "LATITUDE", 'LONGITUDE','NAME', 'REPORT_TYPE', 'CALL_SIGN','TMP','WND','CIG','VIS','DEW','SLP','AA1','AJ1', 'AT1', 'GA1', 'IA1', 'MA1','MD1','OC1','REM']
 
weather_cols = ["STATION", "DATE", "LATITUDE", 'LONGITUDE', 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP', 'AA1', 'AJ1', 'AT1', 'GA1', 'IA1', 'MA1']
 

final_cols = ["DEP_DELAY", "ARR_DELAY",'YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'CRS_DEP_TIME', 'CRS_DEP_TIME_HOUR', 'DEP_TIME_HOUR', 'DEP_DELAY_NEW', 'DEP_TIME_BLK', 'CRS_ARR_TIME', 'CRS_ARR_TIME_HOUR', 'ARR_TIME_HOUR', 'ARR_DELAY_NEW', 'ARR_TIME_BLK', 'DISTANCE', 'DISTANCE_GROUP', 'DEP_DEL15', 'ARR_DEL15', 'ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID', 'CRS_DEP_TIMESTAMP', 'CRS_ARR_TIMESTAMP', 'PR_ARR_DEL15', 'dep_time', 'arr_time', 'FL_PATH', 'ORIGIN_CITY','DEST_CITY',   "CARRIER_DELAY",  "WEATHER_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY",'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN','ACTUAL_ELAPSED_TIME',
'dep_avg_temp_f', 'dep_tot_precip_mm', 'dep_avg_wnd_mps','dep_avg_vis_m', 'dep_avg_slp_hpa', 'dep_avg_dewpt_f', 'arr_avg_temp_f','arr_tot_precip_mm', 'arr_avg_wnd_mps', 'arr_avg_vis_m','arr_avg_slp_hpa', 'arr_avg_dewpt_f']

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta

training_start_date = str(dbutils.widgets.get('01.training_start_date'))
training_end_date = str(dbutils.widgets.get('02.training_end_date'))
inference_date = str(dbutils.widgets.get('03.inference_date'))
airport_code = str(dbutils.widgets.get('00.Airport_Code'))
print(airport_code,training_start_date,training_end_date,inference_date)

inf = dt.date(dt.strptime(str(dbutils.widgets.get('03.inference_date')), "%Y-%m-%d"))
start = dt.date(dt.strptime(str(dbutils.widgets.get('01.training_start_date')), "%Y-%m-%d"))
end = dt.date(dt.strptime(str(dbutils.widgets.get('02.training_end_date')), "%Y-%m-%d"))

# COMMAND ----------

def airlines_transform(dataframe):
  # Selected Columns
  selected_col = ["DEP_DELAY", "ARR_DELAY",
  "YEAR",
  "QUARTER",
  "MONTH",
  "DAY_OF_MONTH",
  "DAY_OF_WEEK",
  "FL_DATE",
  "FL_PATH",
  "OP_UNIQUE_CARRIER",
  "TAIL_NUM",
  "OP_CARRIER_FL_NUM",
  "ORIGIN",
  "ORIGIN_CITY_NAME",
  "ORIGIN_STATE_ABR",
  "DEST",
  "DEST_CITY_NAME",
  "DEST_STATE_ABR",
  "CRS_DEP_TIME",
  "CRS_DEP_TIME_HOUR",
  "DEP_TIME_HOUR",
  "DEP_DELAY_NEW",
  "DEP_TIME_BLK",
  "CRS_ARR_TIME",
  "CRS_ARR_TIME_HOUR",
  "ARR_TIME_HOUR",
  "ARR_DELAY_NEW",
  "CARRIER_DELAY",  
  "WEATHER_DELAY",
  "SECURITY_DELAY",
  "LATE_AIRCRAFT_DELAY",
  'TAXI_OUT', 
  'WHEELS_OFF', 
  'WHEELS_ON', 
  'TAXI_IN',
  'ACTUAL_ELAPSED_TIME',
  "ARR_TIME_BLK",
  "DISTANCE",
  "DISTANCE_GROUP",
  "DEP_DEL15",
  "ARR_DEL15",
  "ORIGIN_AIRPORT_ID",
  "DEST_AIRPORT_ID",
  "CRS_DEP_TIMESTAMP",
  "CRS_ARR_TIMESTAMP",
  "PR_ARR_DEL15"]
  
  # Creating a window partition to extract prior arrival delay for each flight
  windowSpec = Window.partitionBy("TAIL_NUM").orderBy("CRS_DEP_TIMESTAMP")
  
  return (
    dataframe
    .filter("CANCELLED != 1 AND DIVERTED != 1")
    .withColumn("FL_DATE", f.col("FL_DATE").cast("date"))
    .withColumn("OP_CARRIER_FL_NUM", f.col("OP_CARRIER_FL_NUM").cast("string"))
    .withColumn("DEP_TIME_HOUR", dataframe.DEP_TIME_BLK.substr(1, 2).cast("int"))
    .withColumn("ARR_TIME_HOUR", dataframe.ARR_TIME_BLK.substr(1, 2).cast("int"))
    .withColumn("CRS_DEP_TIME_HOUR", f.round((f.col("CRS_DEP_TIME")/100)).cast("int"))
    .withColumn("CRS_ARR_TIME_HOUR", f.round((f.col("CRS_ARR_TIME")/100)).cast("int"))
    .withColumn("DISTANCE_GROUP", f.col("DISTANCE_GROUP").cast("string"))
    .withColumn("OP_CARRIER_FL_NUM", f.concat(f.col("OP_CARRIER"),f.lit("_"),f.col("OP_CARRIER_FL_NUM")))
    .withColumn("FL_PATH", f.concat(f.col("ORIGIN"),f.lit("-"),f.col("DEST")))
    .withColumn("DEP_DEL15", f.col("DEP_DEL15").cast("string"))
    .withColumn("ARR_DEL15", f.col("ARR_DEL15").cast("string"))
    .withColumn("FL_DATE_string", f.col("FL_DATE").cast("string"))
    .withColumn("YEAR", f.col("YEAR").cast("string"))
    .withColumn("QUARTER", f.col("QUARTER").cast("string"))
    .withColumn("MONTH", f.col("MONTH").cast("string"))
    .withColumn("DAY_OF_MONTH", f.col("DAY_OF_MONTH").cast("string"))
    .withColumn("DAY_OF_WEEK", f.col("DAY_OF_WEEK").cast("string"))
    .withColumn("CRS_DEP_TIME_string", f.col("CRS_DEP_TIME").cast("string"))
    .withColumn("CRS_ARR_TIME_string", f.col("CRS_ARR_TIME").cast("string"))
    .withColumn("CRS_DEP_TIME_HOUR_string", f.col("CRS_DEP_TIME_HOUR").cast("string"))
    .withColumn("CRS_ARR_TIME_HOUR_string", f.col("CRS_ARR_TIME_HOUR").cast("string"))
    .withColumn("CRS_DEP_TIME_HH", f.lpad("CRS_DEP_TIME_string", 4, '0').substr(1,2))
    .withColumn("CRS_DEP_TIME_MM", f.lpad("CRS_DEP_TIME_string", 4, '0').substr(3,2))
    .withColumn("CRS_ARR_TIME_HH", f.lpad("CRS_ARR_TIME_string", 4, '0').substr(1,2))
    .withColumn("CRS_ARR_TIME_MM", f.lpad("CRS_ARR_TIME_string", 4, '0').substr(3,2))
    .withColumn("CRS_DEP_TIMESTAMP", f.concat(f.col("FL_DATE_string"),f.lit(" "),f.col("CRS_DEP_TIME_HH"), f.lit(":"),f.col("CRS_DEP_TIME_MM")).cast("timestamp"))
    .withColumn("CRS_ARR_TIMESTAMP", f.concat(f.col("FL_DATE_string"),f.lit(" "),f.col("CRS_ARR_TIME_HH"), f.lit(":"),f.col("CRS_ARR_TIME_MM")).cast("timestamp"))
    .withColumn("CRS_ELAPSED_TIME", f.round((f.col("CRS_ELAPSED_TIME")/60)).cast("int"))
    .withColumn("PR_ARR_DEL15", f.lag(f.col("ARR_DEL15"), 1).over(windowSpec).cast("string"))
    .select(selected_col)
    )


# COMMAND ----------

bronze_airtraffic = spark.sql("select * from dscc202_db.bronze_air_traffic")
bronze_airtraffic = bronze_airtraffic.filter((bronze_airtraffic.DEST == airport_code) | (bronze_airtraffic.ORIGIN == airport_code)).filter(bronze_airtraffic.FL_DATE <= inference_date).filter(bronze_airtraffic.FL_DATE >= training_start_date)


# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
df_airlines = airlines_transform(bronze_airtraffic)
df_airlines = df_airlines.withColumn("dep_time", date_trunc('hour', "CRS_DEP_TIMESTAMP")).withColumn("arr_time", date_trunc('hour', "CRS_ARR_TIMESTAMP"))

# COMMAND ----------

#write to parquet
dbutils.fs.rm(GROUP_DATA_PATH + "/airlines_"+airport_code + ".parquet", recurse=True)
df_airlines.write.parquet(GROUP_DATA_PATH + "/airlines_"+airport_code + ".parquet")


# COMMAND ----------

df_airlines = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "/airlines_"+airport_code + ".parquet")

# COMMAND ----------

df_airlines = df_airlines.filter(df_airlines.FL_DATE <= inf).filter(df_airlines.FL_DATE >= start)
df_airlines.createOrReplaceTempView('airlines')
df_airlines.count()

# COMMAND ----------

df_airlines.count()

# COMMAND ----------

df_airlines.select("FL_DATE").rdd.max()[0]


# COMMAND ----------

inference_date

# COMMAND ----------


from pyspark.sql.functions import *
from pyspark.sql.types import *
df_airlines = df_airlines.withColumn("dep_time", date_trunc('hour', "CRS_DEP_TIMESTAMP")).withColumn("arr_time", date_trunc('hour', "CRS_ARR_TIMESTAMP")).withColumn("FL_PATH", f.concat(f.col("ORIGIN"),f.lit("-"),f.col("DEST"))).withColumn('ORIGIN_CITY', split(col('ORIGIN_CITY_NAME'),",")[0]).withColumn('DEST_CITY', split(col('DEST_CITY_NAME'),",")[0]).withColumn("PR_ARR_DEL15", df_airlines["PR_ARR_DEL15"].cast(IntegerType()))

df_airlines = df_airlines.withColumn('arr_time2',when(df_airlines.arr_time <= df_airlines.dep_time,f.date_add(df_airlines['arr_time'], 1)).otherwise(df_airlines['arr_time'])).withColumn('dep_UNID', f.concat(f.col('dep_time'),f.lit('-'), f.col('ORIGIN'))).withColumn('arr_UNID', f.concat(f.col('arr_time2'),f.lit('-'), f.col('DEST')))

# COMMAND ----------

sqlContext.registerDataFrameAsTable(df_airlines, "silver_airline")
df2 = sqlContext.sql("SELECT DEP_DELAY, ARR_DELAY, YEAR, QUARTER, MONTH, DAY_OF_MONTH, DAY_OF_WEEK, FL_DATE, OP_UNIQUE_CARRIER, TAIL_NUM, OP_CARRIER_FL_NUM, ORIGIN, ORIGIN_CITY_NAME, ORIGIN_STATE_ABR, DEST, DEST_CITY_NAME, DEST_STATE_ABR, CRS_DEP_TIME, CRS_DEP_TIME_HOUR, DEP_TIME_HOUR, DEP_DELAY_NEW, DEP_TIME_BLK, CRS_ARR_TIME, CRS_ARR_TIME_HOUR, ARR_TIME_HOUR, ARR_DELAY_NEW, ARR_TIME_BLK, DISTANCE, DISTANCE_GROUP, DEP_DEL15, ARR_DEL15, ORIGIN_AIRPORT_ID,DEST_AIRPORT_ID, CRS_DEP_TIMESTAMP,TAXI_OUT, WHEELS_OFF, WHEELS_ON, TAXI_IN,ACTUAL_ELAPSED_TIME,  CARRIER_DELAY, WEATHER_DELAY,SECURITY_DELAY,LATE_AIRCRAFT_DELAY,CRS_ARR_TIMESTAMP, PR_ARR_DEL15, dep_time, arr_time2 as arr_time, FL_PATH, ORIGIN_CITY,DEST_CITY, dep_UNID, arr_UNID from silver_airline")

# COMMAND ----------

df_weather=spark.sql("select * from dscc202_group09_db.silver_weather2")
df_weather = df_weather.withColumn('ident', df_weather['CALL_SIGN'].substr(-4, 3)).withColumn('UNID', 
                    f.concat(f.col('TIME'),f.lit('-'), f.col('ident')))

# COMMAND ----------

wtmp = df_weather.select([f.col(c).alias("dep_"+c) for c in df_weather.columns])
df_all = df2.join(wtmp, df2.dep_UNID == wtmp.dep_UNID).drop(wtmp.dep_UNID).drop(wtmp.dep_time)
wtmp = df_weather.select([f.col(c).alias("arr_"+c) for c in df_weather.columns])
df_all = df_all.join(wtmp, df_all.arr_UNID == wtmp.arr_UNID).drop(wtmp.arr_UNID).drop(wtmp.arr_time)

# COMMAND ----------

dbutils.fs.rm(GROUP_DATA_PATH + "/airlines_join_"+ airport_code + ".parquet", recurse=True)
df_all.select(final_cols).write.parquet(GROUP_DATA_PATH + "/airlines_join_"+ airport_code + ".parquet")
df_ready = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "/airlines_join_"+ airport_code + ".parquet")
df_ready.createOrReplaceTempView('airlinestmp')

# COMMAND ----------

dbutils.fs.rm('/mnt/dscc202-group09-datasets/airlines_tmp.delta', recurse=True)

# COMMAND ----------

from pyspark.sql.functions import count, weekofyear
spark.sql("DROP TABLE IF EXISTS airlines_tmp")

group_air_path = GROUP_DATA_PATH + "/airlines_tmp.delta"
sqlCmd2 = """
   CREATE TABLE IF NOT EXISTS {}.airlines_tmp
   USING DELTA
   OPTIONS (path = '{}')
   AS SELECT * from airlinestmp
   """.format(GROUP_DBNAME, group_air_path)
spark.sql(sqlCmd2)
