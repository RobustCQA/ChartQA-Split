# dependencies 
import google.generativeai as genai
import torch 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import time
from eval_utils import modified_relaxed_accuracy

genai.configure(api_key= '')

generation_config = {
  "temperature": 0,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  }
]
model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safety_settings, generation_config=generation_config)
chart_type = "simple"
question_type = "simple"
question_set = "h"

if question_set == "h":
    question_set = "test_human"
else:
    question_set = "test_augmented"

df = pd.read_json('../dataset/{}/{}_{}.json'.format(chart_type, question_set, question_type))
questions = df['query'].tolist()
gold = df['label'].tolist()
imagename = df['imgname'].tolist()
imagename = [f'../dataset/{chart_type}/png/{img}' for img in imagename]
images = [Image.open(img) for img in imagename]
prompt = """You are an expert in answering questions pertaining to a given chart. You will be given a chart and a question. You need to answer the question based on the chart. 
Task: Answer the given question from the chart given to you. 
Instructions: 
1) If a question asks about a column name, give the full and exact name for the column as it is written in the chart. 
2) If a question required multiple outputs, give it in the form: [<output1>, <output2> ..] where outputs are in sorted order. For example, if the output is 'Australia and India' give the answer as [Australia, India]. Please dont use this with column names. 
3) If a question requires doing arithmetic operations, calculate till the final number.
4) If a question asks for what column a certain value is in, give the full and exact name of the column and not the value.
5) If a question asks how many times a certain value appears, give the count and not the name of the columns where it appears.
6) Answer without taking account of the units or scale given in chart. For example, if the chart has values in millions, you should ignore the scale and account absolute numbers. 

You need to carefully look at the whole chart before answering the question directly.
Think step by step and append the answer at the last of your response in the form: "... . The answer is: <answer>"

Question: """

queries = []

for i, question in enumerate(questions):
    text = prompt + question
    queries.append([text, images[i]]) 

model_responses = []

for i in range(0, len(queries)):
    try:
        resp = model.generate_content(queries[i])
        print(i, questions[i], resp.text)
        model_responses.append(resp.text)
    except Exception as e:
        print(i, questions[i], e)
        model_responses.append('Error by gemini')

copy = model_responses.copy()

for i, resp in enumerate(copy):
    if(resp[-1] == '.'):
        resp = resp[:-1]
    if 'The answer is: ' in resp:
        x = resp.split('The answer is: ')
        model_responses[i] = x[1]
    else:
        print("error by gemini")

results = list(zip(questions, model_responses))
final_responses = []
for result in results:
    question, response = result
    final_responses.append(response.strip())

final_responses = [response.split('=')[-1] for response in final_responses]
final_responses = [response.split('%')[0] for response in final_responses]

together = list(zip(final_responses, gold))

model_performance = []

for i, ans in enumerate(final_responses):
    model_score = modified_relaxed_accuracy(questions[i],gold[i], ans)
    # print('gold:', gold[i])
    # print('response:', ans)
    # print('score:', model_score)
    # print()
    model_performance.append(model_score)
    
print('Model accuracy:', sum(model_performance) / len(model_performance))