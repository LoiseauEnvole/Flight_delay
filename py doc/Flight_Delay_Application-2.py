# Databricks notebook source
# MAGIC %md
# MAGIC ## Welcome to The Delay Forecasting Interface
# MAGIC Enter the date of your travel, the airport you are heading to and the time interval you want to set as a training time interval from above window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Me
# MAGIC **Modeling**<br>
# MAGIC Both departure and arrival flights are predicted based upon Random Forest and Gradient Boosting algorithms.<br>
# MAGIC You could find the source code on github at https://github.com/LoiseauEnvole/Flight_delay where files could be imported to Databricks<br>
# MAGIC If you are interested in the detailed resutls and metrics of modeling, please refer The experiment ID: 103614<br>
# MAGIC Feel free to contact us via github or email

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Flight Delay Application
# MAGIC ![Image](https://data-science-at-scale.s3.amazonaws.com/images/Flight+Application+v2.png)
# MAGIC 
# MAGIC The application focus on 13 of the busiest airports in the US.  In particular, 
# MAGIC - JFK - INTERNATIONAL AIRPORT, NY US
# MAGIC - SEA - SEATTLE TACOMA INTERNATIONAL AIRPORT, WA US
# MAGIC - BOS - BOSTON, MA US
# MAGIC - ATL - ATLANTA HARTSFIELD INTERNATIONAL AIRPORT, GA US
# MAGIC - LAX - LOS ANGELES INTERNATIONAL AIRPORT, CA US
# MAGIC - SFO - SAN FRANCISCO INTERNATIONAL AIRPORT, CA US
# MAGIC - DEN - DENVER INTERNATIONAL AIRPORT, CO US
# MAGIC - DFW - DALLAS FORT WORTH AIRPORT, TX US
# MAGIC - ORD - CHICAGO O’HARE INTERNATIONAL AIRPORT, IL US
# MAGIC - CVG - CINCINNATI NORTHERN KENTUCKY INTERNATIONAL AIRPORT, KY US'
# MAGIC - CLT - CHARLOTTE DOUGLAS AIRPORT, NC US
# MAGIC - DCA - WASHINGTON, DC US
# MAGIC - IAH - HOUSTON INTERCONTINENTAL AIRPORT, TX US
# MAGIC 
# MAGIC **The time window starts from**
# MAGIC - January 1st 2015
# MAGIC - Till today

# COMMAND ----------

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

from datetime import datetime
import pandas as pd
import requests
import pandas_profiling
from pandas_profiling.utils.cache import cache_file

# COMMAND ----------

# MAGIC %md
# MAGIC ### Application Widgets
# MAGIC - Airport Code - dropdown list to select the airport of interest.
# MAGIC - Training Start Date - when the training should start (note 6 years of data in the archive)
# MAGIC - Training End Date - when the training should end
# MAGIC - Inference Date - this will be set to 1 day after the training end date although any date after training can be entered.

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta

dbutils.widgets.removeAll()

dbutils.widgets.dropdown("00.Airport_Code", "JFK", ["JFK","SEA","BOS","ATL","LAX","SFO","DEN","DFW","ORD","CVG","CLT","DCA","IAH"])
dbutils.widgets.text('01.training_start_date', "2018-01-01")
dbutils.widgets.text('02.training_end_date', "2019-03-15")
dbutils.widgets.text('03.inference_date', (dt.strptime(str(dbutils.widgets.get('02.training_end_date')), "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"))

training_start_date = str(dbutils.widgets.get('01.training_start_date'))
training_end_date = str(dbutils.widgets.get('02.training_end_date'))
inference_date = str(dbutils.widgets.get('03.inference_date'))
airport_code = str(dbutils.widgets.get('00.Airport_Code'))
print(airport_code,training_start_date,training_end_date,inference_date)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Prep

# COMMAND ----------

import pandas as pd
import json
status = dbutils.notebook.run("ETL", 3600, {"00.Airport_Code":airport_code,"01.training_start_date":training_start_date,"02.training_end_date":training_end_date,"03.inference_date":inference_date})
dbutils.notebook.exit(json.dumps({"exit_code": "Success"}))
if status == "None": print("Passed") 
else: print("Failed")

# COMMAND ----------

#check
df_raw = spark.sql("select * from dscc202_group09_db.airlines_tmp").toPandas()
infer = df_raw.FL_DATE.max()
from datetime import datetime as dt
from datetime import timedelta
df_raw.FL_DATE.max() == dt.date(dt.strptime(str(dbutils.widgets.get('03.inference_date')), "%Y-%m-%d"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modelling & Monitoring
# MAGIC The experiment id is 103614<br>
# MAGIC The model selection is based on squared mean of test error

# COMMAND ----------

import pandas as pd
import json
status = dbutils.notebook.run("Modeling_staging v2", 3600, {"00.Airport_Code":airport_code,"01.training_start_date":training_start_date,"02.training_end_date":training_end_date,"03.inference_date":inference_date})
dbutils.notebook.exit(json.dumps({"exit_code": "Success"}))
if status == "Success" print("Passed") else print("Failed")

# COMMAND ----------

#Backup model - random forest & gradient boosting with 
import pandas as pd
import json
status = dbutils.notebook.run("rf-selene", 3600, {"00.Airport_Code":airport_code,"01.training_start_date":training_start_date,"02.training_end_date":training_end_date,"03.inference_date":inference_date})
dbutils.notebook.exit(json.dumps({"exit_code": "Success"}))
if status == "Success" print("Passed") else print("Failed")

# COMMAND ----------

# run link to the monitoring notebook
status = dbutils.notebook.run("Monitoring_w_pred", 3600, {"00.Airport_Code":airport_code,"01.training_start_date":training_start_date,"02.training_end_date":training_end_date,"03.inference_date":inference_date})
dbutils.notebook.exit(json.dumps({"exit_code": "Success"}))
if status == "Success" print("Passed") else print("Failed")
# NOTE NOTEBOOK SHOULD RETURN dbutils.notebook.exit("Success") WHEN IT PASSES

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction of flights

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta
inf = dt.date(dt.strptime(str(dbutils.widgets.get('03.inference_date')), "%Y-%m-%d"))
inf
#set a time here
import datetime as dt
from datetime import datetime, date
t = datetime.now().time()
TR = dt.datetime.combine(inf, t)
import numpy as np
TR = np.datetime64(TR).astype(datetime)

# COMMAND ----------

import pandas as pd
import json
from pyspark.sql.functions import *
from graphframes import *
from pyspark.sql.functions import lit, col
from pyspark.sql import functions as F
pred_dep = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "/dep_pred_"+ airport_code + ".parquet")
pred_dep.createOrReplaceTempView('dep')
pred_dep = pred_dep.filter(col("ORIGIN").like("%SFO%")).toPandas()
pred_dep['CRS_DEP_TIMESTAMP'] = pd.to_datetime(pred_dep.CRS_DEP_TIMESTAMP)
pred_dep['CRS_ARR_TIMESTAMP'] = pd.to_datetime(pred_dep.CRS_ARR_TIMESTAMP)

pred_dep.loc[(pred_dep['CRS_DEP_TIMESTAMP'] > TR),'Flight Status'] = 'Scheduled'
pred_dep.loc[(pred_dep['CRS_DEP_TIMESTAMP'] < TR)&(pred_dep['CRS_ARR_TIMESTAMP']< TR ), 'Flight Status'] = 'En Route'
pred_dep.loc[pred_dep['CRS_ARR_TIMESTAMP']>= TR , 'Flight Status'] = 'Landed'

pred_dep['dep_pred'] = pred_dep.dep_pred.round()
pred_dep['Estimate'] = pred_dep['CRS_ARR_TIMESTAMP'] + pd.to_timedelta(pred_dep['dep_pred'], unit='m')
import time
pred_dep['Estimate']= pred_dep['Estimate'].apply(lambda x: x.strftime('%H:%M'))
pred_dep.rename(columns={"TAIL_NUM": "Flight_Number", "DEST":"Destination", "DEST_CITY_NAME":"Destination City"}, inplace=True)

display(pred_dep[["Flight_Number","Destination","Destination City",'Flight Status', 'Estimate']])



# COMMAND ----------

pred_arr = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "/arr_pred_"+ airport_code + ".parquet")
pred_arr.createOrReplaceTempView('ARR')
pred_arr = pred_arr.filter(col("DEST").like("%SFO%")).toPandas()
pred_arr['arr_pred'] = pred_arr.arr_pred.astype(int)


# COMMAND ----------


import pandas as pd
import json
from pyspark.sql.functions import *
from graphframes import *
from pyspark.sql.functions import lit, col
from pyspark.sql import functions as F
pred_arr = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "/arr_pred_"+ airport_code + ".parquet")
pred_arr.createOrReplaceTempView('ARR')
pred_arr = pred_arr.filter(col("DEST").like("%SFO%")).toPandas()
pred_arr['CRS_DEP_TIMESTAMP'] = pd.to_datetime(pred_arr.CRS_DEP_TIMESTAMP)
pred_arr['CRS_ARR_TIMESTAMP'] = pd.to_datetime(pred_arr.CRS_ARR_TIMESTAMP)

pred_arr.loc[(pred_arr['CRS_DEP_TIMESTAMP'] > TR),'Flight Status'] = 'Scheduled'
pred_arr.loc[(pred_arr['CRS_DEP_TIMESTAMP'] < TR)&(pred_arr['CRS_ARR_TIMESTAMP']< TR ), 'Flight Status'] = 'En Route'
pred_arr.loc[pred_arr['CRS_ARR_TIMESTAMP']>= TR , 'Flight Status'] = 'Landed'

pred_arr['Estimate'] = pred_arr['CRS_ARR_TIMESTAMP'] + pd.to_timedelta(pred_arr['arr_pred'], unit='m')
import time
pred_arr['Estimate']= pred_arr['Estimate'].apply(lambda x: x.strftime('%H:%M'))
pred_arr.rename(columns={"TAIL_NUM": "Flight_Number", "DEST":"Destination", "DEST_CITY_NAME":"Destination City"}, inplace=True)

display(pred_arr[["Flight_Number","Destination","Destination City",'Flight Status', 'Estimate']])




# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Data Analysis (EDA)

# COMMAND ----------

import pandas as pd
import json
status = dbutils.notebook.run("silver_EDA", 3600, {"00.Airport_Code":airport_code,"01.training_start_date":training_start_date,"02.training_end_date":training_end_date,"03.inference_date":inference_date})
dbutils.notebook.exit(json.dumps({"exit_code": "Success"}))
if status == "None": print("Passed") 
else: print("Failed")
