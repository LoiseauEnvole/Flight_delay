# Databricks notebook source
# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# load the dataset 
dataset_df = spark.sql('select * from dscc202_group09_db.airlines_tmp')

# COMMAND ----------

airlines_tmp = dataset_df

# COMMAND ----------

airlines_tmp.printSchema()

# COMMAND ----------

import numpy as np
import pandas as pd
import json
from pyspark.sql.functions import *
from graphframes import *
from pyspark.sql.functions import lit, col
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md Creating the inference dataframe

# COMMAND ----------

#airlines_tmp_clear=spark.createDataFrame(airlines_tmp_clear)
#airlines_tmp_clear= airlines_tmp_clear.withColumn("dep_date",to_date(col('dep_time')))
#airlines_tmp_clear= airlines_tmp_clear.withColumn("arr_date",to_date(col('arr_time')))

# COMMAND ----------

#display(airlines_tmp_clear)

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta

training_start_date = str(dbutils.widgets.get('01.training_start_date'))
training_end_date = str(dbutils.widgets.get('02.training_end_date'))
inference_date = str(dbutils.widgets.get('03.inference_date'))
airport_code = str(dbutils.widgets.get('00.Airport_Code'))
print(airport_code,training_start_date,training_end_date,inference_date)



# COMMAND ----------


inference_DF =  (airlines_tmp_clear.where(col("FL_DATE") == inference_date))
#display(inference_DF)
inference_DF.count()

# COMMAND ----------

# drop the null values in table 
airlines_tmp_clear = airlines_tmp.dropna(how = 'any')
airlines_tmp_clear.count()
#display(airlines_tmp_clear)

# COMMAND ----------

# MAGIC %md Transforming dataset with selected features to pass through the model

# COMMAND ----------

def clean_df(olddf):
  newdf= olddf.withColumn("CRS_DEP_TIMESTAMP", date_format("CRS_DEP_TIMESTAMP", "yyyy-MM-dd HH:mm")).withColumn("CRS_ARR_TIMESTAMP", date_format("CRS_ARR_TIMESTAMP","yyyy-MM-dd HH:mm")).withColumn("dep_time", date_format("dep_time","yyyy-MM-dd HH:mm")).withColumn("arr_time", date_format("arr_time","yyyy-MM-dd HH:mm"))
  #newdf=newdf.toPandas()
  newdf = newdf.drop(newdf.FL_DATE)
  newdf = newdf.drop(newdf.OP_UNIQUE_CARRIER)
  newdf = newdf.drop(newdf.OP_CARRIER_FL_NUM)
  newdf = newdf[["CRS_DEP_TIME",
                        "CRS_DEP_TIME_HOUR",
                        "CRS_ARR_TIME",
                        "CRS_ARR_TIME_HOUR",
                        "YEAR",
                        "QUARTER",
                        "MONTH",
                        "DAY_OF_MONTH",
                        "DAY_OF_WEEK",
                        "DISTANCE",
                        "DISTANCE_GROUP",
                        "ORIGIN_AIRPORT_ID",
                        "DEST_AIRPORT_ID",
                        "dep_avg_temp_f",
                        "dep_tot_precip_mm",
                        "dep_avg_wnd_mps",
                        "dep_avg_vis_m",
                        "dep_avg_slp_hpa",
                        "dep_avg_dewpt_f",
                        "arr_avg_temp_f",
                        "arr_tot_precip_mm",
                        "arr_avg_wnd_mps",
                        "arr_avg_vis_m",
                        "arr_avg_slp_hpa",
                        "arr_avg_dewpt_f",'ARR_DELAY','DEP_DELAY']]
  return (newdf)

  


# COMMAND ----------

inference_DF1= inference_DF #copy of the dataset with all features for the selected inference date

# COMMAND ----------

inference_DF= clean_df(inference_DF)
airlines_tmp_clear = clean_df(airlines_tmp_clear)

# COMMAND ----------

display(airlines_tmp_clear)

# COMMAND ----------

airlines_tmp_clear.toPandas()

# COMMAND ----------

#extracting x, y 
y1 = airlines_tmp_clear[['ARR_DELAY','DEP_DELAY']]
X = (airlines_tmp_clear.drop(airlines_tmp_clear.ARR_DELAY) and airlines_tmp_clear.drop(airlines_tmp_clear.DEP_DELAY))

#extracting the testing data
inference_DF = (inference_DF.drop(inference_DF.ARR_DELAY) and inference_DF.drop(inference_DF.DEP_DELAY))


# COMMAND ----------

# MAGIC %md Model

# COMMAND ----------

# MAGIC %md The below section includes modeling but,
# MAGIC we need to import the run_id from the model that Richie registered through the Final_modeling.dbc.
# MAGIC I ran the model again just because I was unable to export her model.
# MAGIC Refer to comment in cmd 41.
# MAGIC If registered model is working skip directly to cmd 37 onwards.

# COMMAND ----------

#converting to pandas to pass through modelling functions
X = X.toPandas()
y1 = y1.toPandas()

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y1, random_state=1)

# COMMAND ----------

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

for max_depth in [7,10]:
  for n_estimators in [100,500]:
    for max_features in ["auto","sqrt"]:
      with mlflow.start_run()as run:
          rf = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth)
          rf.fit(X_train, y_train)
          predictions = rf.predict(X_test)

          mlflow.sklearn.log_model(rf, "random-forest-model")
          mlflow.log_params({"n_estimators": n_estimators, "max_features": max_features, "max_depth": max_depth})

          mse = mean_squared_error(y_test, predictions)
          r2 = r2_score(y_test, predictions)
          mlflow.log_metrics({"mse_arr": mse, "r2": r2})

          runID = run.info.run_uuid
          experimentID = run.info.experiment_id

          print(f"Inside MLflow Run with run_id {runID} and experiment_id {experimentID}")


# COMMAND ----------

predictions

# COMMAND ----------

# MAGIC %md Model logging and registry 

# COMMAND ----------

import uuid
model_name = f'airlines_tmp_RFR_model_{uuid.uuid4().hex[:10]}'

# COMMAND ----------

# find the best model 
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from pyspark.sql.functions import min 
runs = mlflow.search_runs()
min_mse = runs['metrics.mse_arr'].min()
best_model_final = runs[runs['metrics.mse_arr']==min_mse]

# COMMAND ----------

#register the best_model
best_model_final_uri = "runs:/{run_id}/random-forest-model".format(run_id=runID)
print (best_model_final_uri)
model_details = mlflow.register_model(model_uri=best_model_final_uri, name=model_name)

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

#Now add a model description
client.update_registered_model(
  name=model_details.name,
  description="This is the very final model forecasts ARR_DELAY&DEP_DELAYbased on various listing inputs."
)

# COMMAND ----------

#Add a version-specific description.
client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using sklearn-RandomForestRegressor."
)

# COMMAND ----------

# MAGIC %md Model staging 

# COMMAND ----------

import mlflow.pyfunc
# if there is model in the staging status
#model_name = "sk-learn-random-forest-reg-model"
#stage = 'Staging'

#model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

#model.predict(data)

# COMMAND ----------

inference_DF =inference_DF.toPandas()

# COMMAND ----------

def output(data, model_uri):
    model = mlflow.sklearn.load_model(model_uri)
    return model.predict(data)

# COMMAND ----------

# Formulate the model URI- fetch from the model registery
model_uri = best_model_final_uri
#model_uri = 'airlines_tmp_RFR_model_25cd21f3fc'  #the registered model goes in here

# Predict the Power output 
pred_production = pd.DataFrame(output(inference_DF, model_uri), columns=[["predicted_arr","predicted_dep"]], index=inference_DF.index)
pred_production.reset_index(level=[0])
pred_production.columns

# COMMAND ----------

class S_Model(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        pred = self.model.predict(model_input.values)
        return pred

# COMMAND ----------

#Save model
final_model_path =  f"{GROUP_DATA_PATH}/final-random-forest-model-arr-"+airport_code
dbutils.fs.rm(workingDir+'/final-random-forest-model-arr-'+airport_code, recurse=True)
mlflow.pyfunc.save_model(path=final_model_path, python_model=S_Model(model = mlflow.pyfunc.load_model(best_model_final_uri)))


# COMMAND ----------

# MAGIC %md Pushing outcome to production

# COMMAND ----------

#combining with actual dataset
y1["predicted_arr"]=  pred_production.iloc[:, 0]
y1["predicted_dep"]= pred_production.iloc[:, 1]
y1

# COMMAND ----------

# MAGIC %md the NaN in the above outcome represent the dates outside of the inference date (for which the prediction has been made)

# COMMAND ----------

# MAGIC %md Plot of the delays per hour of the inference date

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import matplotlib.pyplot as plt
# MAGIC pred_production.plot.line(figsize=(11,8))
# MAGIC plt.title("Delay Plot")
# MAGIC plt.ylabel("Minutes of delay")
# MAGIC plt.xlabel("No. of delays")

# COMMAND ----------

# MAGIC %md Output dataframe

# COMMAND ----------

inference_DF1=inference_DF1.toPandas()

# COMMAND ----------

outcols = ['TAIL_NUM','OP_CARRIER_FL_NUM','DEST','DEST_CITY_NAME','CRS_DEP_TIMESTAMP']
dep_out = pd.concat([inference_DF1[outcols].reset_index(drop=True), pd.DataFrame(pred_production.iloc[:, 1])], axis=1)
dep_out.columns = ['TAIL_NUM','OP_CARRIER_FL_NUM','DEST','DEST_CITY_NAME','CRS_DEP_TIMESTAMP','Predicted_depDelay']
dep_out #Predicted_depDelay is the delay in minutes?


# COMMAND ----------

outcols = ['TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN','ORIGIN_CITY_NAME','CRS_ARR_TIMESTAMP']
arr_out = pd.concat([inference_DF1[outcols].reset_index(drop=True), pd.DataFrame(pred_production.iloc[:, 0])], axis=1)
arr_out.columns = ['TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN','ORIGIN_CITY_NAME','CRS_ARR_TIMESTAMP','Predicted_arrDelay']
arr_out

# COMMAND ----------

# MAGIC %md 
# MAGIC The dataframes to be used in the application are: dep_out, arr_out

# COMMAND ----------

dbutils.fs.rm(GROUP_DATA_PATH + "/f_arr_pred_"+ airport_code + ".parquet", recurse=True)
arr_out.to_parquet("/dbfs"+ GROUP_DATA_PATH + "/f_arr_pred_"+ airport_code + ".parquet")

# COMMAND ----------

dbutils.fs.rm(GROUP_DATA_PATH + "/f_dep_pred_"+ airport_code + ".parquet", recurse=True)
dep_out.to_parquet("/dbfs"+ GROUP_DATA_PATH + "/f_dep_pred_"+ airport_code + ".parquet")
