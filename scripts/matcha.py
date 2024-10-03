import pandas as pd
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
from eval_utils import modified_relaxed_accuracy

processor = Pix2StructProcessor.from_pretrained('google/matcha-chartqa', device_map='cuda')
model = Pix2StructForConditionalGeneration.from_pretrained('google/matcha-chartqa', device_map='cuda')

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
images = [Image.open(img) for img in images]

model_responses = []

for i, question in enumerate(questions):
    inputs = processor(images=images[i], text=question, return_tensors="pt")
    inputs = inputs.to('cuda')
    predictions = model.generate(**inputs, max_new_tokens=512)
    response = processor.decode(predictions[0], skip_special_tokens=True)
    print(response)
    model_responses.append(response)

final_responses = [model_response.split('%')[0].strip() for model_response in model_responses]

model_performance = []

for i, ans in enumerate(final_responses):
    model_score = modified_relaxed_accuracy(questions[i], gold[i], ans)
    # print('Response:', ans)
    # print('Gold:', gold[i])
    # print('Model score:', model_score)
    # print()
    model_performance.append(model_score)

print('Model accuracy:', sum(model_performance) / len(model_performance))