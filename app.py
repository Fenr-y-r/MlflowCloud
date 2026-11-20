import sys
import logging
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow 
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()
remote_server_uri=os.getenv("remote_server_uri")

logging.basicConfig(level=logging.WARN)
logger=logging.getLogger(__name__)

def eval_metrics(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error(actual,pred)
    r2=r2_score(actual,pred)
    return rmse,mae,r2

if __name__=="__main__":
    ##Reading dataset
    try:
        data=pd.read_csv("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv",sep=";")
    except Exception:
        logger.exception("Can't find data")
    train,test=train_test_split(data) #default is 75/25
    
    train_x=train.drop(["quality"],axis=1)
    test_x=test.drop(["quality"],axis=1)
    train_y=train["quality"]
    test_y=test["quality"]
    
    alpha=float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    mlflow.set_tracking_uri(remote_server_uri)
    url_type=urlparse(mlflow.get_tracking_uri()).scheme
    
    with mlflow.start_run():
        lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
        lr.fit(train_x,train_y)
        predicted_qualities=lr.predict(test_x)
        (rmse,mae,r2)=eval_metrics(test_y,predicted_qualities)
        print("{:f}, {:f}".format(alpha,l1_ratio))
        print(f"{rmse},{mae},{r2}")    
        
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("r2",r2)
        mlflow.log_metric("mae",mae)    
        
        if url_type!="file":
            mlflow.sklearn.log_model(lr,"model",registered_model_name="ElasticNetWineMode")
        else:
            mlflow.sklearn.log_model(lr,"model")