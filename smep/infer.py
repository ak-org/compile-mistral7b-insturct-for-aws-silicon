import json
import boto3
from botocore.config import Config
import time 
import sys 
from transformers import AutoTokenizer
import array
        
def run_infer(endpoint_name, body):
    resp = smr.invoke_endpoint(EndpointName=endpoint_name,
                                Body=body,
                                ContentType="application/json")
    results = resp['Body'].read().decode(errors="ignore")
    return results

def encode_text(tokenizer, prompt):
    encoded_text = tokenizer(prompt, padding=True, return_tensors="pt")
    tokens_length = encoded_text.input_ids.shape[1]
    return encoded_text, tokens_length


def mistral_inference_calls(prompt, endpoint_name):
    body = {
            "inputs": f"""[INST]{prompt}[/INST]""",
            "parameters": {'n_positions': 1024, 'top_p': 0.9, 'temperature': 0.9}
        }
    start = time.time()
    results = run_infer(endpoint_name, json.dumps(body).encode('utf-8'))
    duration = time.time() - start
    print(f'\nPrediction took {duration} seconds\n')
    results_text = json.loads(results)
    results = results_text['generated_text'][len(body["inputs"]):]
    return duration, body["inputs"], results
    
def generate_results(endpoint_name, tokenizer, prompt):
    duration, input_prompt, results = mistral_inference_calls(prompt, endpoint_name=endpoint_name)
    encoded_input_prompt_text, input_tokens_count = encode_text(tokenizer, input_prompt)
    print('Input tokens length:', input_tokens_count)
    encoded_results_text, results_tokens_count = encode_text(tokenizer, results)
    print('Results tokens length:', results_tokens_count)
    return duration, input_prompt, results, \
           input_tokens_count, results_tokens_count

if __name__ == "__main__":
    long_prompt = "Artificial intelligence (AI) has become a fundamental part of our daily lives, shaping industries and redefining the way we work, play, and interact with the world. From speech recognition and image analysis to autonomous vehicles and personalized recommendations, AI technologies are transforming various aspects of human society.\n\nAI, at its core, refers to the simulation of human intelligence processes by machines, especially computer systems. These processes include learning (the acquisition of information and rules for using the information), reasoning (using rules to reach approximate or definite conclusions), and self-correction. The field of AI research was founded on the claim that human intelligence can be so precisely described that a machine can be made to simulate it.\n\nSince its inception in the mid-20th century, the field of AI has undergone significant advancements and has seen its share of ups and downs. In the early decades, AI research focused on symbolic methods, rule-based systems, and expert systems. However, these methods faced limitations and failed to meet the high expectations set by early predictions.\n\nThe rise of machine learning, a subset of AI that involves the development of algorithms that allow computers to learn from and make decisions or predictions based on data, marked a significant turning point in the field. Machine learning, and more recently, deep learning, which models high-level abstractions in data using multiple processing layers, have enabled groundbreaking applications in various domains, such as computer vision, natural language processing, and robotics.\n\nHowever, while the impact of AI is undeniable, it also raises important ethical, societal, and technical challenges. These include concerns about job displacement due to automation, privacy issues related to data collection and use, algorithmic bias, the transparency and explainability of AI systems, and the potential misuse of AI technology.\n\nTaking into consideration this context, what are the fundamentals of AI and its various branches such as machine learning and deep learning? How has the field of AI evolved over time, and what are its major applications in various domains? What are some of the challenges and concerns associated with the widespread adoption of AI, and how can they be addressed?"
    short_prompt = "What is your favorite condiment?"
    medium_prompt = "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy, usually from the sun, into chemical energy in the form of glucose. This process is crucial for life on Earth as it provides the oxygen we breathe and contributes to the food we consume. It also plays a significant role in the global carbon cycle, helping to mitigate climate change. How does photosynthesis contribute to sustaining life on Earth and its role in mitigating climate change?"
    boto_config = Config(
        read_timeout=900, # wait for 15 mins for large prompts
        connect_timeout=60
    )
    boto3_session=boto3.session.Session(region_name="us-east-1")
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1',
                                              padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    smr = boto3.client('sagemaker-runtime',
                       config=boto_config)
    # Change the value to reflect endpoint name in your env
    if len(sys.argv) != 2:
        print('Error: Specified inference endpoint')
        exit(-1)
    else:
        endpoint_name = sys.argv[1]
        r1 = open('benchmark.csv', 'w')
        r2 = open('results.txt', 'w') 
        r1.write('duration,input_token_count,results_token_count\n')
        r2.write('prompt|results\n')
        for prompt_file in ['short_prompts.txt', 'medium_prompts.txt']:
            with open(prompt_file, 'r')as f:
                prompts = f.readlines()
                for prompt in prompts:
                    duration, \
                    input_prompt, \
                    results, \
                    input_tokens_count, \
                    results_tokens_count = generate_results(endpoint_name, 
                                                            tokenizer, 
                                                            prompt.replace('\n', ''))
                    results = results.replace('\n', '')
                    r1.write(f"{duration},{input_tokens_count},{results_tokens_count}\n")
                    r2.write(f"{input_prompt}|{results}\n")
        r1.close()
        r2.close()
