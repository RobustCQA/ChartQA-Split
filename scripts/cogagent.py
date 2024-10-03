import torch 
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoModelForCausalLM, LlamaTokenizer
from PIL import Image
import json

MODEL_PATH = "THUDM/cogagent-vqa-hf"
TOKENIZER_PATH = "lmsys/vicuna-7b-v1.5"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
torch_type = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    cache_dir = "./media/vivek/c33fd89b-a307-4208-a045-64d021572535/models_cache",
    torch_dtype=torch_type,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map = "auto"
).to(DEVICE).eval()

def ask_cog_agent(image_path, query, history=[], temperature=0, do_sample=False):
    """
    Get the response from the cogagent based on the given image, query, and conversation history.

    Parameters:
    - image_path (str): Path to the image file.
    - query (str): The current query prompt.
    - history (list): List of tuples containing the conversation history.
    - temperature (float): Sampling temperature for response generation (default is 0.9).
    - do_sample (bool): Whether to use sampling during response generation (default is False).

    Returns:
    - str: The generated response from the cogagent.
    """

    image1 = Image.open(image_path).convert('RGB')
    image2 = Image.open(image_path).convert('RGB')

    input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

    gen_kwargs = {"max_length": 4096, "temperature": temperature, "do_sample": do_sample}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("</s>")[0]
    return response

type = ["augmented", "human"]
chart_type = ["simple", "complex"]
ques_type = ["simple", "complex"]

prompt = """You will be given a question. Reason step by step and mention your answer."""

for i in range(2):
    for j in range(2):
        for k in range(2):
            df = pd.read_json(f'../dataset/{chart_type[i]}/test_{type[j]}_{ques_type[k]}.json')
            questions = df['query'].tolist()
            gold = df['label'].tolist()
            imagename = df['imgname'].tolist()
            imagename = [f'../dataset/{chart_type[i]}/png/{img}' for img in imagename]
            model_responses = []
            for L in range(0, len(questions)):
                response = ask_cog_agent(imagename[L], prompt + questions[L])
                model_responses.append(response)
            with open(f'../cogagent_results/responses_{chart_type[i]}_{type[j]}_{ques_type[k]}.json', 'w') as f:
                json.dump(model_responses, f)
            del model_responses
            del df
            del questions
            del gold
            del imagename
            torch.cuda.empty_cache()
            

            