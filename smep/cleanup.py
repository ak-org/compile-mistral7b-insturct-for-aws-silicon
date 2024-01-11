import boto3
import sagemaker
from sagemaker import Model
import os
from datetime import datetime
import sys 

aws_region='us-east-1'
boto3_session=boto3.session.Session(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'], 
    aws_session_token=os.environ['AWS_SESSION_TOKEN'],
    region_name=aws_region)
sm = boto3.client('sagemaker')

sm.delete_model(ModelName=sys.argv[1])
sm.delete_endpoint_config(EndpointConfigName=sys.argv[1])
sm.delete_endpoint(EndpointName=sys.argv[1])