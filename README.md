### MLFLOW on Azure

Originally this project was made for AWS however i repurprosed it for Azure instead due to constraints 

This is a simple ElasticNet ML model for grading wine quality but with hosting the mlflow tracking server on an azure VM (instead of an ec2 instance) and storing the model outputs on an azure storage blob (instead of s3)

The model is trained locally and then sent to the VM for viewing which then stores it in the blob

