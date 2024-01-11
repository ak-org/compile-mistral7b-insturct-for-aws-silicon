import boto3
import os
import sys


if __name__ == "__main__":
    ENDPOINT_MAME = sys.argv[1]
    boto3_session=boto3.session.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'], 
        aws_session_token=os.environ['AWS_SESSION_TOKEN'],
        region_name="us-east-1")
    smr = boto3.client('sagemaker-runtime')
    sm = boto3.client('sagemaker')
    response = sm.describe_endpoint(EndpointName=ENDPOINT_MAME)
    print(f"Status of Endpoint {response['EndpointName']}")
    print(f"{response['EndpointStatus']}")
    