import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import json
torch.set_grad_enabled(False)

ckpt_path = 'internlm/internlm-xcomposer2-vl-7b'
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt_path, cache_dir = "/media/vivek/c33fd89b-a307-4208-a045-64d021572535/models_cache", device_map="cuda", trust_remote_code=True).eval().cuda().half()
model.tokenizer = tokenizer

def askInternLM(prompt, question, image_path):
    query = f'<ImageHere>{prompt} {question}'
    image = f'{image_path}'
    with torch.cuda.amp.autocast():
        response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
    return response

type = ["augmented", "human"]
chart_type = ["simple", "complex"]
ques_type = ["simple", "complex"]

prompt  = """You will be given a chart and a question pertaining to it. Explain your answer, and at the last of your response, append in the form: "... . The answer is: <answer>". Think step by step and make sure you reach the correct output.
Question: """

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
                response = askInternLM(prompt, questions[L], imagename[L])
                model_responses.append(response)
            with open(f'../internLM_results/responses_{chart_type[i]}_{type[j]}_{ques_type[k]}.json', 'w') as f:
                json.dump(model_responses, f)
            del model_responses
            del df
            del questions
            del gold
            del imagename
            torch.cuda.empty_cache()
            