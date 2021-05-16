# Databricks notebook source
# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

#Imports, widgets and the like
from pyspark.sql.functions import *
from pyspark.sql.types import *

from sklearn.model_selection import train_test_split
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import get_statistics_html
import warnings

spark.conf.set("spark.sql.shuffle.partitions", "32")
spark.conf.set("spark.sql.adaptive.enabled", "true")

spark.sql(f"USE dscc202_group09_db")

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from datetime import datetime as dt
from datetime import timedelta


training_start_date = str(dbutils.widgets.get('01.training_start_date'))
training_end_date = str(dbutils.widgets.get('02.training_end_date'))
inference_date = str(dbutils.widgets.get('03.inference_date'))
airport_code = str(dbutils.widgets.get('00.Airport_Code'))

# COMMAND ----------

#Bronze airport data
train_query = f"""select fl_date, CRS_DEP_TIME, CRS_ARR_TIME, YEAR, QUARTER, MONTH, DAY_OF_MONTH, DAY_OF_WEEK, DISTANCE, DISTANCE_GROUP, ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID, ARR_DELAY, DEP_DELAY
from dscc202_db.bronze_air_traffic
where fl_date between '{training_start_date}' and '{training_end_date}'"""

serve_query = f"""select fl_date, CRS_DEP_TIME, CRS_ARR_TIME, YEAR, QUARTER, MONTH, DAY_OF_MONTH, DAY_OF_WEEK, DISTANCE, DISTANCE_GROUP, ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID
from dscc202_db.bronze_air_traffic
where fl_date  between '{training_end_date}' and '{inference_date}'"""

train_df = spark.sql(train_query).sample(withReplacement=False, fraction=0.001).toPandas()
serve_df = spark.sql(serve_query).sample(withReplacement=False, fraction=0.1).toPandas()

# COMMAND ----------

display(train_df)

# COMMAND ----------

#Bronze weather data
weather_train_query = f"""select STATION, date, LATITUDE, LONGITUDE, NAME, REPORT_TYPE, CALL_SIGN, WND, CIG, VIS, TMP, DEW, SLP, AA1, AJ1, AT1, GA1, IA1, MA1, MD1, OC1, REM
from dscc202_db.bronze_weather
where STRING(DATE(date)) between '{training_start_date}' and '{training_end_date}'"""

weather_serve_query = f"""select STATION, date, LATITUDE, LONGITUDE, NAME, REPORT_TYPE, CALL_SIGN, WND, CIG, VIS, TMP, DEW, SLP, AA1, AJ1, AT1, GA1, IA1, MA1, MD1, OC1, REM
from dscc202_db.bronze_weather
where STRING(DATE(date)) between '{training_end_date}' and '{inference_date}'"""

weather_train_df = (spark.sql(weather_train_query)
        .withColumn('temp_f', split(col('TMP'),",")[0]*9/50+32)
        .withColumn('temp_qual', split(col('TMP'),",")[1])
        .withColumn('wnd_deg', split(col('WND'),",")[0])
        .withColumn('wnd_1', split(col('WND'),",")[1])
        .withColumn('wnd_2', split(col('WND'),",")[2])
        .withColumn('wnd_mps', split(col('WND'),",")[3]/10)
        .withColumn('wnd_4', split(col('WND'),",")[4])
        .withColumn('vis_m', split(col('VIS'),",")[0])
        .withColumn('vis_1', split(col('VIS'),",")[1])
        .withColumn('vis_2', split(col('VIS'),",")[2])
        .withColumn('vis_3', split(col('VIS'),",")[3])
        .withColumn('dew_pt_f', split(col('DEW'),",")[0]*9/50+32)
        .withColumn('dew_1', split(col('DEW'),",")[1])
        .withColumn('slp_hpa', split(col('SLP'),",")[0]/10)
        .withColumn('slp_1', split(col('SLP'),",")[1])
        .withColumn('precip_hr_dur', split(col('AA1'),",")[0])
        .withColumn('precip_mm_intvl', split(col('AA1'),",")[1]/10)
        .withColumn('precip_cond', split(col('AA1'),",")[2])
        .withColumn('precip_qual', split(col('AA1'),",")[3])
        .withColumn('precip_mm', col('precip_mm_intvl')/col('precip_hr_dur'))
        .withColumn("time", date_trunc('hour', "DATE"))
        .where("REPORT_TYPE='FM-15' and NAME LIKE '% AIRPORT%'")
        .groupby("time")
        .agg(mean('temp_f').alias('avg_temp_f'), \
             sum('precip_mm').alias('tot_precip_mm'), \
             mean('wnd_mps').alias('avg_wnd_mps'), \
             mean('vis_m').alias('avg_vis_m'),  \
             mean('slp_hpa').alias('avg_slp_hpa'),  \
             mean('dew_pt_f').alias('avg_dewpt_f'), )
        .sample(withReplacement=False, fraction=0.01).toPandas() )
weather_serve_df = (spark.sql(weather_serve_query)
        .withColumn('temp_f', split(col('TMP'),",")[0]*9/50+32)
        .withColumn('temp_qual', split(col('TMP'),",")[1])
        .withColumn('wnd_deg', split(col('WND'),",")[0])
        .withColumn('wnd_1', split(col('WND'),",")[1])
        .withColumn('wnd_2', split(col('WND'),",")[2])
        .withColumn('wnd_mps', split(col('WND'),",")[3]/10)
        .withColumn('wnd_4', split(col('WND'),",")[4])
        .withColumn('vis_m', split(col('VIS'),",")[0])
        .withColumn('vis_1', split(col('VIS'),",")[1])
        .withColumn('vis_2', split(col('VIS'),",")[2])
        .withColumn('vis_3', split(col('VIS'),",")[3])
        .withColumn('dew_pt_f', split(col('DEW'),",")[0]*9/50+32)
        .withColumn('dew_1', split(col('DEW'),",")[1])
        .withColumn('slp_hpa', split(col('SLP'),",")[0]/10)
        .withColumn('slp_1', split(col('SLP'),",")[1])
        .withColumn('precip_hr_dur', split(col('AA1'),",")[0])
        .withColumn('precip_mm_intvl', split(col('AA1'),",")[1]/10)
        .withColumn('precip_cond', split(col('AA1'),",")[2])
        .withColumn('precip_qual', split(col('AA1'),",")[3])
        .withColumn('precip_mm', col('precip_mm_intvl')/col('precip_hr_dur'))
        .withColumn("time", date_trunc('hour', "DATE"))
        .where("REPORT_TYPE='FM-15' and NAME LIKE '% AIRPORT%'")
        .groupby("time")
        .agg(mean('temp_f').alias('avg_temp_f'), \
             sum('precip_mm').alias('tot_precip_mm'), \
             mean('wnd_mps').alias('avg_wnd_mps'), \
             mean('vis_m').alias('avg_vis_m'),  \
             mean('slp_hpa').alias('avg_slp_hpa'),  \
             mean('dew_pt_f').alias('avg_dewpt_f'), ).toPandas() )

# COMMAND ----------

#Get stats and schema
air_train_stats = tfdv.generate_statistics_from_dataframe(dataframe=train_df)
air_schema = tfdv.infer_schema(statistics=air_train_stats)
air_serve_stats = tfdv.generate_statistics_from_dataframe(dataframe=serve_df)

weather_train_stats = tfdv.generate_statistics_from_dataframe(dataframe=weather_train_df)
weather_schema = tfdv.infer_schema(statistics=weather_train_stats)
weather_serve_stats = tfdv.generate_statistics_from_dataframe(dataframe=weather_serve_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bronze Data Analysis and Anomaly and Drift Detection:
# MAGIC Here, we visualize differences in our training and serving data - here, serving data consists of the time between the training end date and the inference data. Note that we shouldn't worry about the discrepancy in the target variables (ARR_DELAY and DEP_DELAY) due to the nature of serving date, nor should we worry about the discrepancy in any of the date variables (YEAR, QUARTER, MONTH, DAY_OF_MONTH, DAY_OF_WEEK, FL_DATE), as these are obviously directly dependent on the exact dates we sample - so long as the serving data at least comes from the same distribution as the training data, we should not need to retrain our model. The same applies to weather data, though less so - weather data sampled over the course of a day or so (which is the period of time taken in by our serving data) is, naturally, going to come from a smaller range of values than weather sampled over several years; again, so long as the serving data at least comes from the same distribution as the training data, we should not need to retrain our model. If the other variables (CRS_DEP_TIME, CRS_ARR_TIME, DISTANCE, DISTANCE_GROUP, ORIGIN_AIRPORT_ID) appear misaligned, or if the date or weather serving data began clearly coming from different distributions (for example, if we suddenly started taking in values for 1980 for YEAR), then retraining would be in order. We can formalize this with a goodness-of-fit test; however, a visual examination of the histograms is sufficient for now.

# COMMAND ----------

#Comparing airline serving data with training data
displayHTML(get_statistics_html(lhs_statistics=air_serve_stats, rhs_statistics=air_train_stats,
                          lhs_name='AIR_SERVE_DATASET', rhs_name='AIR_TRAIN_DATASET'))

# COMMAND ----------

#Comparing weather serving data with training data
displayHTML(get_statistics_html(lhs_statistics=weather_serve_stats, rhs_statistics=weather_train_stats,
                          lhs_name='WEATHER_SERVE_DATASET', rhs_name='WEATHER_TRAIN_DATASET'))

# COMMAND ----------

# MAGIC %md
# MAGIC We also examine anomalies within our dataset; while no outstanding anomalies of note have been detected during our spoofed serving data, we would likely want to retrain if, for example, we suddenly added or gained a variable (which would become apparent here). Less potent anomalies (for example, a sudden up-tick in null values) would likely not require retraining - though, issues such as this are subjective, and would be taken on a case-by-case basis.

# COMMAND ----------

# Check anomalies for airline data
print("Training Schema Check (Weather):")
schema_anomalies = tfdv.validate_statistics(statistics=air_train_stats, schema=air_schema)
tfdv.display_anomalies(schema_anomalies)

air_schema.default_environment.append('AIR_TRAINING')
air_schema.default_environment.append('AIR_SERVING')

# Specify that delay features are not in AIR_SERVING environment.
tfdv.get_feature(air_schema, 'DEP_DELAY').not_in_environment.append('AIR_SERVING')
tfdv.get_feature(air_schema, 'ARR_DELAY').not_in_environment.append('AIR_SERVING')

serving_anomalies_with_env = tfdv.validate_statistics(air_serve_stats, air_schema, environment='AIR_SERVING')

print("Serving Schema Check (Airline):")
tfdv.display_anomalies(serving_anomalies_with_env)

# COMMAND ----------

#Check anomalies for weather data
print("Training Schema Check (Weather):")
schema_anomalies = tfdv.validate_statistics(statistics=weather_train_stats, schema=weather_schema)
tfdv.display_anomalies(schema_anomalies)

#weather_schema.default_environment.append('WEATHER_TRAINING')
#weather_schema.default_environment.append('WEATHER_SERVING')

#tot_precip_mm = tfdv.get_feature(weather_schema, 'tot_precip_mm')
#tot_precip_mm.distribution_constraints.min_domain_mass = 0.9

print("Serving Schema Check (Weather):")
weather_serve_schema_anomalies = tfdv.validate_statistics(statistics=weather_serve_stats, schema=weather_schema)
tfdv.display_anomalies(weather_serve_schema_anomalies)

# COMMAND ----------

## load df inference
df_raw = spark.sql("select * from dscc202_group09_db.airlines_tmp").toPandas()
infer = df_raw.FL_DATE.max()
df_inf = df_raw[df_raw.FL_DATE == infer]
display(df_inf)

# COMMAND ----------

df_inf.count()
[df_inf[i].fillna(0, inplace=True) for i in num_feat]

# COMMAND ----------

label_col = 'ARR_DELAY'
num_feat = ["DISTANCE",'TAXI_OUT', 'WHEELS_OFF','WHEELS_ON','TAXI_IN',"PR_ARR_DEL15",'ACTUAL_ELAPSED_TIME',
            "CARRIER_DELAY","WEATHER_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY","DEP_DELAY_NEW",
            'dep_avg_temp_f','dep_tot_precip_mm', 'dep_avg_wnd_mps', 'dep_avg_vis_m','dep_avg_slp_hpa', 'dep_avg_dewpt_f', 'arr_avg_temp_f','arr_tot_precip_mm', 'arr_avg_wnd_mps', 'arr_avg_vis_m','arr_avg_slp_hpa', 'arr_avg_dewpt_f']
cat_feat = []

num_feat_dep = ["DISTANCE",'TAXI_OUT', 'WHEELS_OFF','WHEELS_ON','TAXI_IN',"PR_ARR_DEL15",'ACTUAL_ELAPSED_TIME',
            "CARRIER_DELAY","WEATHER_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY","ARR_DELAY_NEW",
            'dep_avg_temp_f','dep_tot_precip_mm', 'dep_avg_wnd_mps', 'dep_avg_vis_m','dep_avg_slp_hpa', 'dep_avg_dewpt_f', 'arr_avg_temp_f','arr_tot_precip_mm', 'arr_avg_wnd_mps', 'arr_avg_vis_m','arr_avg_slp_hpa', 'arr_avg_dewpt_f']
label_col_dep = 'DEP_DELAY'

# COMMAND ----------

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import numpy as np
X_arr_infer = minmax_scale(df_inf[num_feat])
X_dep_infer = minmax_scale(df_inf[num_feat_dep])

X_arr_infer[np.isnan(X_arr_infer)] = 0
X_dep_infer[np.isnan(X_dep_infer)] = 0

# COMMAND ----------

import mlflow.pyfunc

model_name = "gb_model_arrival45c7301a2e"
stage = 'Staging'

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{stage}"
)

arr_pred = model.predict(X_arr_infer)

# COMMAND ----------

import mlflow.pyfunc

model_name = "random_forest_departure32ee15e306"
stage = 'Staging'

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{stage}"
)

dep_pred = model.predict(X_dep_infer)

# COMMAND ----------

outcols = ['TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN','ORIGIN_CITY_NAME','DEST','DEST_CITY_NAME','CRS_DEP_TIMESTAMP', 'CRS_ARR_TIMESTAMP']
dep_out = pd.concat([df_inf[outcols].reset_index(drop=True), pd.DataFrame(dep_pred)], axis=1)
dep_out.rename(columns={0: "dep_pred"}, inplace=True)

# COMMAND ----------

outcols = ['TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN','ORIGIN_CITY_NAME','DEST','DEST_CITY_NAME','CRS_DEP_TIMESTAMP', 'CRS_ARR_TIMESTAMP']
arr_out = pd.concat([df_inf[outcols].reset_index(drop=True), pd.DataFrame(arr_pred)], axis=1)
arr_out.rename(columns={0: "arr_pred"}, inplace=True)

# COMMAND ----------

#write predictions
dbutils.fs.rm(GROUP_DATA_PATH + "/arr_pred_"+ airport_code + ".parquet", recurse=True)
arr_out.to_parquet("/dbfs"+ GROUP_DATA_PATH + "/arr_pred_"+ airport_code + ".parquet")
dbutils.fs.rm(GROUP_DATA_PATH + "/dep_pred_"+ airport_code + ".parquet", recurse=True)
dep_out.to_parquet("/dbfs"+ GROUP_DATA_PATH + "/dep_pred_"+ airport_code + ".parquet")

# COMMAND ----------

## load predictions
pred_dep = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "/dep_pred_"+ airport_code + ".parquet")
pred_arr = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "/arr_pred_"+ airport_code + ".parquet")

# COMMAND ----------

display(pred_dep) #dep_pred is the estimates 

# COMMAND ----------

display(pred_arr) #arr_pred is the estimates 

# COMMAND ----------

#check 
pred_arr.count() == len(df_inf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### "Promoting" to Production
# MAGIC Finally, we illustrate how we would promote a staged, retrained model - the function promoteCheck() below compares a staged model to the model in production, and if we deem it necessary, the function promote() promotes the staged model in question to production. 

# COMMAND ----------

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
import mlflow

#score_model is a helper function here
def score_model(data, model_uri):
  model = mlflow.sklearn.load_model(model_uri)
  return model.predict(data)

#Checks a new, staged model against production (if we needed to)
def promoteCheck(df_inf):
  score_df = df_inf
  actual_arr_delay = pd.DataFrame(score_df.power.values, columns=['ARR_DELAY'], index=score_df.index)
  actual_dep_delay = pd.DataFrame(score_df.power.values, columns=['DEP_DELAY'], index=score_df.index)
  score = score_df.drop("ARR_DELAY", axis=1).drop("DEP_DELAY", axis=1)
  
  model_uri_production = "models:/{}/{}".format('sk-learn-random-forest-reg-model', 'Production')
  pred_production = pd.DataFrame(score_model(score, model_uri_production), columns=["arr_delay", "dep_delay"], index=score_df.index)
  
  model_uri_staging = "models:/{}/{}".format('sk-learn-random-forest-reg-model', 'Staging')
  pred_staging = pd.DataFrame(score_model(score, model_uri_staging), columns=["arr_delay", "dep_delay"], index=score_df.index)
  
  actual_arr_delay["prod_arr_delay"] = pred_production["arr_delay"]
  actual_dep_delay["prod_dep_delay"] = pred_production["dep_delay"]
  actual_arr_delay["stage_arr_delay"] = pred_staging["arr_delay"]
  actual_dep_delay["stage_dep_delay"] = pred_staging["dep_delay"]
  
  #Print graphics
  %matplotlib inline
  actual_arr_delay.plot.line(figsize=(11,8))
  plt.title("Production and Staging Model Comparison (Arrivals)")
  plt.ylabel("Minutes of Delay")
  plt.xlabel("Time")
  
  %matplotlib inline
  actual_dep_delay.plot.line(figsize=(11,8))
  plt.title("Production and Staging Model Comparison (Delays)")
  plt.ylabel("Minutes of Delay")
  plt.xlabel("Time")

def promote(model_uri):
  model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
  client = MlflowClient()
  
  client.update_registered_model(
  name=model_details.name,
  description="This is our retrained, new model.")
  
  client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Production')
  
