from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import pandas as pd
from eval_utils import modified_relaxed_accuracy

model_name = "ahmed-masry/unichart-chartqa-960"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

df = pd.read_json('../dataset/simple/test_augmented_complex.json')

questions = df['query'].tolist()
gold = df['label'].tolist()
images = df['imgname'].tolist()

prompt = "Give an exact answer to the question:"

images = [f'../dataset/simple/png/{img}' for img in images]
images = [Image.open(img).convert("RGB") for img in images]
questions = [f"<chartqa> {q} <s_answer>" for q in questions]

batch_size = 8

batched_ques = [questions[i:i+batch_size] for i in range(0, len(questions), batch_size)]
batched_imgs = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]
n_batches = len(batched_ques)
model_responses = []

for i in range(len(questions)):
    decoder_input_ids = processor.tokenizer(questions[i], add_special_tokens=False, return_tensors="pt", padding = True).input_ids
    pixel_values = processor(images[i], return_tensors="pt").pixel_values
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = sequence.split("<s_answer>")[1].strip()
    model_responses.append(sequence)
    print(i, questions[i], sequence)
    
final_responses = [model_response.split('%')[0].strip() for model_response in model_responses]

from typing import Optional
import re
  
model_performance = []

for i, ans in enumerate(final_responses):
    model_score = modified_relaxed_accuracy(questions[i], gold[i], ans)
    # print('Response:', ans)
    # print('Gold:', gold[i])
    # print('Model score:', model_score)
    # print()
    model_performance.append(model_score)

print('Model accuracy:', sum(model_performance) / len(model_performance))