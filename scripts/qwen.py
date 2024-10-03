import torch 
import pandas as pd
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_utils import modified_relaxed_accuracy

model_dir = snapshot_download('qwen/Qwen-VL')

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cuda",
    trust_remote_code=True
).eval()

tokenizer.pad_token = '<|endoftext|>'
tokenizer.padding_side = 'left'

chart_type = "complex"
question_type = "simple"
question_set = "a"

if question_set == "h":
    question_set = "test_human"
else:
    question_set = "test_augmented"

df = pd.read_json('../dataset/{}/{}_{}.json'.format(chart_type, question_set, question_type))

questions = df['query'].tolist()
gold = df['label'].tolist()
images = df['imgname'].tolist()
img_path = '../dataset/{}'.format(chart_type)
images = [img_path + '/png/' + img for img in images]

queries = []

for i, question in enumerate(questions):
    text = question + ' Answer:'
    queries.append(tokenizer.from_list_format([
            {'image': images[i]},
            {'text': text},
    ])) 

# batch the queries
batch_size = 8

batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]

model_responses = []

for batch in batches:
    inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
    inputs.to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(generated)
    model_responses.extend(generated)

# zip the question and the model response
results = list(zip(questions, model_responses))
final_responses = []

# extract final response
for result in results:
    question, response = result
    # final_responses.append(response.split(question)[-1].strip())
    new_response = response.split('Answer:')[-1].strip()
    new_response = new_response.split('%')[0].strip()
    
    final_responses.append(new_response)

model_performance = []

for i, ans in enumerate(final_responses):
    model_score = modified_relaxed_accuracy(questions[i], gold[i], ans)
    # print('Response:', ans)
    # print('Gold:', gold[i])
    # print('Model score:', model_score)
    # print()
    model_performance.append(model_score)

print('Model accuracy:', sum(model_performance) / len(model_performance))