mkdir mymodel
cp serving-gpu.properties  mymodel/serving.properties
tar czvf mymodel-gpu.tar.gz mymodel/
rm -rf mymodel
aws s3 cp mymodel-gpu.tar.gz s3://gai-model-artifacts/lmi/Mistral-7B-Instruct-v0.1/code/