import io
import json
import boto3
import os
from datetime import datetime
import time 
import sys 
import pprint

def run_infer2(endpoint_name, body):
    resp = smr.invoke_endpoint(EndpointName=endpoint_name,
                                Body=body,
                                ContentType="application/json")
    #print('\n\nResponse is', resp['Body'].read().decode(errors="ignore"))
    #results = resp['Body'].read().decode(errors="ignore")
    results = resp['Body'].read().decode(errors="ignore")
    return results

def new_inference_calls(endpoint_name):
    body = {
            "inputs": "[INST]Explain the concept of Generative AI to me as if I am an 8 year old child. [/INST]",
            "parameters": {
                'do_sample': 'true',
                'max_new_tokens': 128, 
                'top_k':50, 
                'temperature': 0.9
            }
        }
    start = time.time()
    results = run_infer2(endpoint_name, json.dumps(body).encode('utf-8'))
    end = time.time()
    print(f'\nPrediction took {end-start} seconds\n')
    print(f'\nThis is the result of inference request #1 \n\n {results}')
    
def new_inference_calls_batch(endpoint_name):
    body = {
            "inputs": "[INST]tell me a story of the little red riding hood. [/INST]",
            "parameters": {
                'do_sample': 'true',
                'max_new_tokens': 1024, 
                'top_p': 0.9,
                "top_k": 49,
                'temperature': 0.9
            }
        }
    print(body)
    start = time.time()
    results = run_infer2(endpoint_name, json.dumps(body).encode('utf-8'))
    end = time.time()
    print(f'\nPrediction took {end-start} seconds\n')
    #print(f'\nThis is the result of inference request #1 \n\n {results}')
    result_json = json.loads(results)
    print(json.dumps(result_json, indent=2))
        
if __name__ == "__main__":
    boto3_session=boto3.session.Session(region_name="us-east-1")
    smr = boto3.client('sagemaker-runtime')
    # Change the value to reflect endpoint name in your env
    if len(sys.argv) != 2:
        print('Error: Specified inference endpoint')
        exit(-1)
    else:
        endpoint_name = sys.argv[1]
        new_inference_calls(endpoint_name=endpoint_name)
