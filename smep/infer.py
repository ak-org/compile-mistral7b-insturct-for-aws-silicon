import io
import json
import boto3
import os
from datetime import datetime
import time 
import sys 

        
def run_infer(endpoint_name, body):
    resp = smr.invoke_endpoint(EndpointName=endpoint_name,
                                                    Body=body,
                                                    ContentType="application/json")
    results = resp['Body'].read().decode(errors="ignore")
    return results


def new_inference_calls(endpoint_name):
    body = {
            "inputs": """[INST] Can you write me a poem about steamed hams?[/INST]""",
            "parameters": {'n_positions': 320, 'top_p': 0.9, 'temperature': 0.9}
        }
    start = time.time()
    results = run_infer(endpoint_name, json.dumps(body).encode('utf-8'))
    end = time.time()
    print(f'\nPrediction took {end-start} seconds\n')
    print(f'\nThis is the result of inference request #1 \n\n {results}')
    
    
if __name__ == "__main__":
    boto3_session=boto3.session.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'], 
        aws_session_token=os.environ['AWS_SESSION_TOKEN'],
        region_name="us-east-1")
    smr = boto3.client('sagemaker-runtime')
    # Change the value to reflect endpoint name in your env
    if len(sys.argv) != 2:
        print('Error: Specified inference endpoint')
        exit(-1)
    else:
        endpoint_name = sys.argv[1]
        new_inference_calls(endpoint_name=endpoint_name)
