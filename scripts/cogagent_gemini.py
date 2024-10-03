# dependencies
import google.generativeai as genai
import pandas as pd
import json
import time
from eval_utils import modified_relaxed_accuracy

genai.configure(api_key='')

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
model = genai.GenerativeModel(
    'gemini-1.5-flash', safety_settings=safety_settings, generation_config=generation_config)

prompt = """You are an expert in getting the answers from a given long answer with steps. These questions were asked about a chart.
Task: Extract the final answer based on the given long sequence of reasoning with answer, given the question.
 
Instructions:
Append to your response and reasoning: 'The answer is: <final_answer>'. 

If a question asks about a column name, give the full and exact name for the column as it is written in answer. 
If a question required multiple outputs and the output contains multiple outputs as well, give it in the form: [<output1>, <output2> ..] where outputs are in sorted order. For example, if the output is 'Australia and India' give the answer as [Australia, India]. 
Ignore percentage signs.
Remove the units from the answer. For example, if the answer is '10 million', give the answer as '10'. 

A few examples:

Question: What is the value of the blue column?
Given Answer: The blue column has the name 'XXX' and the value is 10.
Your Answer: <reasoning>. The answer is: 10

Question: What is the share of people above 65+ years in the small business category?
Given Answer: To find the share of SME owners in small business over 65 years, we need to add the percentages for the '65-69 years' and '70-74 years' age groups. The calculation is as follows: 26.1% (65-69 years) + 11.8% (70-74 years) = 37.9%. So, the share of SME owners in small business over 65 years is 37.9%.
Your Answer: <reasoning>. The answer is: 37.9

Where <reasoning>. is your reasoning and your chain of thought to get to the answer.

You need to carefully look at the question and the given answer. Think step by step.

Question: {question}
Given Answer: {answer}"""

c = 'complex'
t = 'human'
q = 'complex'


df = pd.read_json(f'../dataset/{c}/test_{t}_{q}.json')
questions = df['query'].tolist()
gold = df['label'].tolist()
pred = json.load(open('../cogagent_results/' +
                 f'responses_{c}_{t}_{q}.json', 'r'))
model_responses = []

assert (len(pred) == len(questions))

for i in range(len(questions)):
    time.sleep(0.5)
    try:
        resp = model.generate_content(prompt.format(
            question=questions[i], answer=pred[i])).text
        model_responses.append(resp)
        print(i, resp)
    except Exception as e:
        print(e)
        model_responses.append('error')

fin = []
for i, resp in enumerate(model_responses):
    if (resp[-1] == '.'):
        resp = resp[:-1]
    if 'The answer is: ' in resp:
        x = resp.split('The answer is: ')
        fin.append(x[1])
    else:
        fin.append('error')
        print("error by gemini")
results = list(zip(questions, fin))
final_responses = []
for result in results:
    question, response = result
    final_responses.append(response.strip())
together = list(zip(final_responses, gold))
model_performance = []
for i, ans in enumerate(final_responses):
    model_score = modified_relaxed_accuracy(questions[i], gold[i], ans)
    model_performance.append(model_score)

print("Model performance: ", sum(model_performance), "out of", len(
    model_performance), ">>", sum(model_performance)/len(model_performance))
