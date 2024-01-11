import boto3
import sagemaker
from sagemaker import Model
import os

MODEL_NAME="smep-inf2-mistral-7b-instruct"
aws_region='us-east-1'
model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
boto3_session=boto3.session.Session()
    #aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
    #aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'], 
    #aws_session_token=os.environ['AWS_SESSION_TOKEN'],
    #region_name=aws_region
smr = boto3.client('sagemaker-runtime')
sm = boto3.client('sagemaker')
role = 'arn:aws:iam::102048127330:role/service-role/SageMaker-ak-datascientist'  # execution role for the endpoint
#role = sagemaker.get_execution_role()
instance_type = "ml.inf2.8xlarge"
endpoint_name = sagemaker.utils.name_from_base(MODEL_NAME)

sess = sagemaker.session.Session(boto3_session, 
                                 sagemaker_client=sm, 
                                 sagemaker_runtime_client=smr)  # sagemaker session for interacting with different AWS APIs
account = sess.account_id()  # account_id of the current SageMaker Studio environment
bucket_name = 'gai-model-artifacts'
release='2.16.0'
prefix=f'torchserve/{release}'
output_path = f"s3://{bucket_name}/{prefix}"

print(f'account={account}, region={aws_region}, role={role}, output_path={output_path}')
s3_uri = f'{output_path}/model_store/{model_id}/' #  "s3://sagemaker-us-east-1-102048127330/torchserve/model_store/mistralai/Mistral-7B-Instruct-v0.1/"
print("======================================")
print(f'Will load artifacts from {s3_uri}')
print("======================================")
image_uri = f'102048127330.dkr.ecr.{aws_region}.amazonaws.com/neuronx-torch2:{release}'
print("======================================")
print(f'Using Container image {image_uri}')
print("======================================")

model = Model(
    name=endpoint_name,
    # Enable SageMaker uncompressed model artifacts
    model_data={
        "S3DataSource": {
                "S3Uri": s3_uri,
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
        }
    },
    image_uri=image_uri,
    role=role,
    sagemaker_session=sess,
    env={"TS_INSTALL_PY_DEP_PER_MODEL": "true"},
)
print(model)

model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    endpoint_name=endpoint_name,
    volume_size=512, # increase the size to store large model
    model_data_download_timeout=3600, # increase the timeout to download large model
    container_startup_health_check_timeout=600, # increase the timeout to load large model,
    wait=False,
)

print(f'\n Model deployment initiated. \n{endpoint_name}\n')