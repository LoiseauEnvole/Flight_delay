# Databricks notebook source
# MAGIC %md
# MAGIC This is a backup version of modeling for improving RMSE and for monitoring to start early.<br>
# MAGIC The experiment ID is 103614

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %scala
# MAGIC dbutils.notebook.getContext.notebookPath

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta

training_start_date = str(dbutils.widgets.get('01.training_start_date'))
training_end_date = str(dbutils.widgets.get('02.training_end_date'))
inference_date = str(dbutils.widgets.get('03.inference_date'))
airport_code = str(dbutils.widgets.get('00.Airport_Code'))
print(airport_code,training_start_date,training_end_date,inference_date)


# COMMAND ----------

from datetime import datetime
start = pd.to_datetime(training_start_date,format= '%Y/%m/%d')
end = pd.to_datetime(training_end_date,format= '%Y/%m/%d')
infer = pd.to_datetime(inference_date,format= '%Y/%m/%d')

# COMMAND ----------

import pandas as pd
df_raw = spark.sql("select * from dscc202_group09_db.airlines_tmp").toPandas()
#df_raw['FL_DATE'] = pd.to_datetime(df_raw['FL_DATE'], format= '%Y/%m/%d')


# COMMAND ----------

df_raw.FL_DATE.min()
df_raw.FL_DATE.max()

# COMMAND ----------

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df_raw.dtypes


# COMMAND ----------

display(df_raw)

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

[df_raw[i].fillna(0, inplace=True) for i in num_feat]

df_model = df_raw[df_raw.FL_DATE < infer]
df_topred = df_raw[df_raw.FL_DATE == infer]


# COMMAND ----------

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import numpy as np
X_train, X_test, y_train, y_test = train_test_split(df_model[num_feat], df_model[[label_col]].values.ravel(), random_state=99)

X_train = minmax_scale(X_train)
X_test = minmax_scale(X_test)

np.where(X_train >= np.finfo(np.float64).max)

# COMMAND ----------

X_infer = minmax_scale(df_topred[num_feat])
len(X_train) #192081
X_train = X_train[~np.isnan(X_train).any(axis=1)]
len(X_train)
X_test = X_test[~np.isnan(X_test).any(axis=1)]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Arrival

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import uuid
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from mlflow.models.signature import infer_signature
from sklearn import metrics



# COMMAND ----------

#mlflow.set_experiment('/Users/wbao@ur.rochester.edu/flight_delay/G09_test')

# COMMAND ----------

import mlflow.pyfunc
from  mlflow.tracking import MlflowClient
client = MlflowClient()
#client.delete_registered_model(name="SFO_model_arrival35d026a73c")

# COMMAND ----------

# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow.sklearn
import json
#mlflow.set_experiment('/Users/wbao@ur.rochester.edu/flight_delay/G09_test')
for n in [100, 200]:
  for d in [15,25]:
      with mlflow.start_run(run_name="RF Model Arr") as run:
        n_estimators = n
        max_features = "sqrt"
        max_depth = 25
        dataSet = "Alt"
        rf = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)

        mlflow.sklearn.log_model(rf, "random-forest-arr")
        mlflow.log_params({"n_estimators": n_estimators, "max_features": max_features, "max_depth": max_depth, "dataset": dataSet})

        mse =np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mlflow.log_metrics({"mse_arr": mse, "r2_arr": r2})

        runID = run.info.run_uuid
        experimentID = run.info.experiment_id
        model_name = f"gb_model_arrival{uuid.uuid4().hex[:10]}"
        model_uri = "runs:/{run_id}/model".format(run_id=runID)

        print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

print(model_uri)
print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

import mlflow.pyfunc
from  mlflow.tracking import MlflowClient
client = MlflowClient()

client = MlflowClient()
runs = client.search_runs(experimentID, order_by=["metrics.mse_arr"], max_results=1)
runs[0]
runID = runs[0].info.run_uuid
runID

# COMMAND ----------

#Gradient Boosting Regression experimentID103614
#mlflow.set_experiment('/Users/wbao@ur.rochester.edu/flight_delay/G09_test')
from sklearn.ensemble import GradientBoostingRegressor
for n in [200, 300, 500]:
  for d in [10,20]
    with mlflow.start_run(run_name="Boosting Model") as run:
      n_estimators = n
      max_features = "sqrt"
      max_depth = d
      dataSet = "Unreduced"
      rf = GradientBoostingRegressor(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth)
      rf.fit(X_train, y_train)
      predictions = rf.predict(X_test)

      mlflow.sklearn.log_model(rf, "gbr-arr")
      mlflow.log_params({"n_estimators": n_estimators, "max_features": max_features, "max_depth": max_depth, "dataset": dataSet})

      mse = np.sqrt(mean_squared_error(y_test, predictions))
      r2 = r2_score(y_test, predictions)
      mlflow.log_metrics({"mse_arr": mse, "r2_arr": r2})
      runID = run.info.run_uuid
      experimentID = run.info.experiment_id
      model_name = f"gb_model_arrival{uuid.uuid4().hex[:10]}"
      model_uri = "runs:/{run_id}/model".format(run_id=runID)
      print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

import mlflow.pyfunc
from  mlflow.tracking import MlflowClient
client = MlflowClient()

runs = client.search_runs(experimentID, order_by=["metrics.mse_arr"], max_results=1)
runs[0]
runID = runs[0].info.run_uuid
runID

# COMMAND ----------

#Register the best one with lowest rmse
model_name = f"{airport_code}_model_arrival{uuid.uuid4().hex[:10]}"
model_uri = "runs:/{run_id}/model".format(run_id=runID)
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

artifactURI = runs[0].info.artifact_uri + "/random-forest-arr"
artifactURI

# COMMAND ----------

class Arr_Model(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        pred = self.model.predict(model_input.values)
        return pred

# COMMAND ----------

workingDir= 'dbfs:/Users/wbao@ur.rochester.edu/flight_delay/rf_selene'
working_path = workingDir.replace("dbfs:", "/dbfs")
final_model_path =  f"{working_path}/final-model-arr-"+airport_code
dbutils.fs.rm(workingDir+'/final-model-arr-'+airport_code, recurse=True)
mlflow.pyfunc.save_model(path=final_model_path, python_model=Arr_Model(model = mlflow.pyfunc.load_model(artifactURI)))

# COMMAND ----------

len(df_topred)

# COMMAND ----------

python_function = mlflow.pyfunc.load_model(final_model_path)
arr_pred = python_function.predict(pd.DataFrame(X_infer))#pyfunc's predictions
print(mlflow.pyfunc.load_model(artifactURI).predict(X_infer)) #price predictions directly from the RF model


# COMMAND ----------

outcols = ['TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN','ORIGIN_CITY_NAME','DEST','DEST_CITY_NAME','CRS_DEP_TIMESTAMP', 'CRS_ARR_TIMESTAMP']
arr_out = pd.concat([df_topred[outcols].reset_index(drop=True), pd.DataFrame(arr_pred)], axis=1)
arr_out.rename(columns={0: "arr_pred"}, inplace=True)

# COMMAND ----------

display(arr_out)

# COMMAND ----------

dbutils.fs.rm(GROUP_DATA_PATH + "/arr_pred_"+ airport_code + ".parquet", recurse=True)
arr_out.to_parquet("/dbfs"+ GROUP_DATA_PATH + "/arr_pred_"+ airport_code + ".parquet")

# COMMAND ----------

def deviance_plot(est, X_test, y_test, ax=None, label='', train_color='#2c7bb6', 
                  test_color='#d7191c', alpha=1.0, ylim=(0, 10)):
    """Deviance plot for ``est``, use ``X_test`` and ``y_test`` for test error. """
    n_estimators = len(est.estimators_)
    test_dev = np.empty(n_estimators)

    for i, pred in enumerate(est.staged_predict(X_test)):
       test_dev[i] = est.loss_(y_test, pred)

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        
    ax.plot(np.arange(n_estimators) + 1, test_dev, color=test_color, label='Test %s' % label, 
             linewidth=2, alpha=alpha)
    ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color, 
             label='Train %s' % label, linewidth=2, alpha=alpha)
    ax.set_ylabel('Error')
    ax.set_xlabel('n_estimators')
    ax.set_ylim(ylim)
    return test_dev, ax

# COMMAND ----------

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
#deviance_plot(rf, X_test, y_test, ylim=(0,0.3))
#plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Departure

# COMMAND ----------

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_model[num_feat_dep], df_model[[label_col_dep]].values.ravel(), random_state=99)

X_train = minmax_scale(X_train)
X_test = minmax_scale(X_test)

np.where(X_train >= np.finfo(np.float64).max)

# COMMAND ----------

X_infer = minmax_scale(df_topred[num_feat_dep])

len(X_train) #192081
X_train = X_train[~np.isnan(X_train).any(axis=1)]
len(X_train)
X_test = X_test[~np.isnan(X_test).any(axis=1)]

# COMMAND ----------

#cross validation - took more than 1 hour

#from sklearn.model_selection import GridSearchCV
#param_grid = {'learning_rate': [0.05, 0.02],'max_depth': [6, 12],'min_samples_leaf': [5, 6],'max_features': [0.2, 0.6]}

#gb = GradientBoostingRegressor(n_estimators=600, loss='huber')
#gb_cv = GridSearchCV(gb, param_grid, cv=3, n_jobs=-1)
#gb_cv.fit(X_train, y_train)
#gb_cv.best_estimator_

# COMMAND ----------

#mean_absolute_error(ytest, gb_cv.predict(Xtest))

# COMMAND ----------

#deviance_plot(gb_cv.best_estimator_, X_test, y_test, ylim=(0,40))
#plt.legend()

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# Random Forest Regressor experimentID 103614
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow.sklearn
mlflow.set_experiment('/Users/wbao@ur.rochester.edu/flight_delay/G09_test')
with mlflow.start_run(run_name="RF Model Dep") as run:
  n_estimators = 100
  max_features = "sqrt"
  max_depth = 25
  dataSet = "Alt"
  rf = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth)
  rf.fit(X_train, y_train)
  predictions = rf.predict(X_test)
  
  mlflow.sklearn.log_model(rf, "rf-dep")
  mlflow.log_params({"n_estimators": n_estimators, "max_features": max_features, "max_depth": max_depth, "dataset": dataSet})
  
  mse =np.sqrt(mean_squared_error(y_test, predictions))
  r2 = r2_score(y_test, predictions)
  mlflow.log_metrics({"mse_dep": mse, "r2_dep": r2})
  
  runID = run.info.run_uuid
  experimentID = run.info.experiment_id
  
  model_name = f"rf_model_departure{uuid.uuid4().hex[:10]}"
  model_uri = "runs:/{run_id}/model".format(run_id=runID)
  print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

#Gradient Boosting Regression experimentID 103614
mlflow.set_experiment('/Users/wbao@ur.rochester.edu/flight_delay/G09_test')
from sklearn.ensemble import GradientBoostingRegressor
for n in [100,200]:
  for d in [15,25]:
    with mlflow.start_run(run_name="Boosting Model") as run:
      n_estimators = n
      max_features = "sqrt"
      max_depth = d
      dataSet = "Unreduced"
      rf = GradientBoostingRegressor(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth)
      rf.fit(X_train, y_train)
      predictions = rf.predict(X_test)

      mlflow.sklearn.log_model(rf, "gbr-dep")
      mlflow.log_params({"n_estimators": n_estimators, "max_features": max_features, "max_depth": max_depth, "dataset": dataSet})

      mse = np.sqrt(mean_squared_error(y_test, predictions))
      r2 = r2_score(y_test, predictions)
      mlflow.log_metrics({"mse_dep": mse, "r2_dep": r2})

      runID = run.info.run_uuid
      experimentID = run.info.experiment_id

      print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

#deviance_plot(rf, X_test, y_test, ylim=(0,40));
#plt.legend()

# COMMAND ----------

import mlflow.pyfunc
from  mlflow.tracking import MlflowClient
client = MlflowClient()

client = MlflowClient()
runs = client.search_runs(experimentID, order_by=["metrics.mse_dep"], max_results=1)
runID = runs[0].info.run_uuid
runID

# COMMAND ----------

 runs[0].info.artifact_uri

# COMMAND ----------

#Register the best one with lowest rmse
model_name = f"{airport_code}_model_departure{uuid.uuid4().hex[:10]}"
model_uri = "runs:/{run_id}/model".format(run_id=runID)
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

artifactURI_dep = runs[0].info.artifact_uri + "/gbr-dep"
artifactURI_dep

# COMMAND ----------



# COMMAND ----------

class Dep_Model(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        pred = self.model.predict(model_input.values)
        return pred

# COMMAND ----------

workingDir= 'dbfs:/Users/wbao@ur.rochester.edu/flight_delay/rf_selene'
working_path = workingDir.replace("dbfs:", "/dbfs")
final_model_path =  f"{working_path}/final-model-dep-"+airport_code
dbutils.fs.rm(workingDir+'/final-model-dep-'+airport_code, recurse=True)


# COMMAND ----------

mlflow.pyfunc.save_model(path=final_model_path, python_model=Arr_Model(model = mlflow.pyfunc.load_model(artifactURI_dep)))

# COMMAND ----------

python_function = mlflow.pyfunc.load_model(final_model_path)
dep_pred = python_function.predict(pd.DataFrame(X_infer))#pyfunc's predictions
print(mlflow.pyfunc.load_model(artifactURI_dep).predict(X_infer)) #price predictions directly from the RF model

# COMMAND ----------

outcols = ['TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN','ORIGIN_CITY_NAME','DEST','DEST_CITY_NAME','CRS_DEP_TIMESTAMP', 'CRS_ARR_TIMESTAMP']
dep_out = pd.concat([df_topred[outcols].reset_index(drop=True), pd.DataFrame(dep_pred)], axis=1)
dep_out.rename(columns={0: "dep_pred"}, inplace=True)

# COMMAND ----------

dbutils.fs.rm(GROUP_DATA_PATH + "/dep_pred_"+ airport_code + ".parquet", recurse=True)
dep_out.to_parquet("/dbfs"+ GROUP_DATA_PATH + "/dep_pred_"+ airport_code + ".parquet")

# COMMAND ----------

#Register the best one with lowest rmse
model_name = f"{airport_code}_model_departure_{runID.hex[:10]}"
model_uri = "runs:/{run_id}/model".format(run_id=runID)
#model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # Polishing models with Grid Search
# MAGIC Note that took more than 40 minutes

# COMMAND ----------

##### Polishing models if time permit
y = df_raw[[label_col_dep]]
X = df_raw[num_feat_dep]
X = minmax_scale(X)
from sklearn.model_selection import train_test_split
gbx_train, gbx_test, gby_train, gby_test = train_test_split(X,y,test_size = 0.30, random_state = 42)


# COMMAND ----------


from sklearn.impute import KNNImputer
 
imputer = KNNImputer(n_neighbors=3) 
from pyspark.ml.regression import GBTRegressor 
gbt = GBTRegressor(labelCol=label_col)

# Redefining X to be the imputed data

import numpy as np
import pandas as pd
import mlflow
import uuid
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from mlflow.models.signature import infer_signature
from sklearn import metrics


#mlflow.end_run()
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return (rmse, mae, r2)

def objective(params):
    # Create a gradient boosting regression model
    gbt = GradientBoostingRegressor(n_estimators=int(params['n_estimators']),max_depth=int(params['max_depth']))
    # Use the cross-validation accuracy to compare the models' performance
    error = cross_val_score(gbt, X, y).mean()
    return {'loss': error, 'status': STATUS_OK}

def flight_gbr(X_train, y_train, X_test, y_test, group = 'arr'):
  """
  Returns xgb regressor for sparse data base
  -------
  experimentID
  runID
  reg: our GB model
  
  """
  search_space = {'n_estimators': hp.quniform('n_estimators', 100,500,100), 'max_depth':  hp.quniform('max_depth', 5,15,5)}
 
  spark_trials = SparkTrials(parallelism=36)
 
  # Turning on autolog
  mlflow.sklearn.autolog()
 
  # I modified the logging to log the numpy array instead of the pandas dataframe since
  # X_train, y_train, are numpy arrays
  # Here, we use a context manager to help ML flow run the code
  mlflow.set_experiment('/Users/wbao@ur.rochester.edu/flight_delay/G09_test')
  with mlflow.start_run(run_name="Gradient Boosting Regressor") as run:
  
    # fmin allows us to use the bayesian grid search method and set a few other params
    best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=32, trials=spark_trials, rstate=np.random.RandomState(99))
 
    # Run model
    reg = GradientBoostingRegressor(max_depth=int(best_params['max_depth']), n_estimators=int(best_params['n_estimators']), random_state=0)
    reg.fit(X_train, y_train)
 
    # Log model
    mlflow.sklearn.log_model(reg, "gb-model", input_example=X[:5], signature=infer_signature(X, y)) 
    
    # Getting vector of predictions
    y_pred = reg.predict(X_test)
 
    # Root Mean Squared Error Calculation
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(f"rmse: {rmse}")
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, y_pred)
    # Log metrics
    mlflow.log_metrics({"mse_"+group : rmse, "r2"+group : r2})
 
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
    
    # Registering my model
    model_name = f"{group}_gb_model_{uuid.uuid4().hex[:10]}"
    print(f"model name: {model_name}")
    model_uri = "runs:/{run_id}/model".format(run_id=runID)
    print(f"model uri: {model_uri}")
    #model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"model details: {model_details}")
 
    print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")
 
  # Print the best value found for number of estimators and the max depth of the tree
  print("Best value found: ", best_params)
  
  # Return the best model
  return experimentID, runID, reg, model_uri

# COMMAND ----------

experimentID, runID, xgb_reg, model_uri = flight_gbr(gbx_train, gby_train, gbx_test, gby_test, group='dep')

# COMMAND ----------

### XGBOOST IS THE BEST BUT TOOK LONG
import click
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
import xgboost as xgb
import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return (rmse, mae, r2)


@click.command()
@click.option("--training_data")
@click.option("--test_data")
@click.option("--label_col")
@click.option("--max_depth", default=7)
@click.option("--ntrees", default=200)
@click.option("--lr", default=0.005)
def main(training_data, test_data, label_col, max_depth, ntrees, lr):


    xgbRegressor = xgb.XGBRegressor(
        max_depth=max_depth,
        n_estimators=ntrees,
        learning_rate=lr,
        random_state=42,
        seed=42,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=1,
        gamma=1)
    pipeline = Pipeline(steps=[("regressor", xgbRegressor)])

    pipeline.fit(XTrain, yTrain)
    yPred = pipeline.predict(XTest)
    
    (rmse, mae, r2) = eval_metrics(yTest, yPred)
    
    print("XGBoost tree model (max_depth=%f, trees=%f, lr=%f):" % (max_depth, ntrees, lr))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("ntrees", ntrees)
    mlflow.log_param("lr", lr)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("MAE", mae)
    
    mlflow.sklearn.log_model(pipeline, "model")
    #print("Model saved in run %s" % mlflow.active_run_id())




