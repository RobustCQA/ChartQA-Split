from openai import OpenAI
from glob import glob 
import os
import json

client = OpenAI(api_key = "Enter API Key Here")

batch_ids = {}

chart_types = ["complex", "simple"]
ques_types = ["complex", "simple"]
gen_types = ["human", "augmented"]

for chart_type in chart_types:
    for ques_type in ques_types: 
        for gen_type in gen_types:
          print(f"Creating batch for {chart_type}_{ques_type}_{gen_type}")
          batch_input_file = client.files.create(
            file=open(f"./GPT_batches/{chart_type}/{ques_type}_{gen_types}.jsonl", "rb"),
            purpose="batch"
          )
          batch = client.batches.create(
            input_file_id = batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
              "description": f"{chart_type}_{ques_type}_{gen_type}"
            }
          )
          batch_ids[f"{chart_type}_{ques_type}_{gen_type}"] = batch.id
          print(f"Batch created for {(chart_type, ques_type, gen_type)} with id: {batch.id}")

json.dump(batch_ids, open("batch_ids.json", "w"))