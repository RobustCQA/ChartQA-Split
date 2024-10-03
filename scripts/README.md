# **Scripts for running Question-Answering on ChartQA-Split**

## **Below mentioned are the files for running the Question-Answering on ChartQA-Split**

```
└── scripts
    ├──cogagent.py
    ├──cogagent_gemini.py
    ├──deplot-extract.py
    ├──deplot-flan.py
    ├──eval_utils.py
    ├──gemini.py
    ├──GPT_batch_creation.py
    ├──GPT_batch_running.py
    ├──GPT_batch_saving.py
    ├──GPT_results.py
    ├──intern_gemini.py
    ├──internLM.py
    ├──matcha.py
    ├──qwen.py
    ├──unichart.py
    ├──README.md
```

### **General Information about the files:**

1. We have provided both interactive notebooks and scripts for the Question-Answering on ChartQA-Split.

2. The files are named according to the model they are running. For example, `cogagent.py` is the script for running the cogagent model and `cogagent_gemini.py` is the script for pipelining the results of the cogagent model to the gemini model.

3. For each model, you can choose to change the type of questions, charts (*simple and complex*) and generation types (*human and augmented*) that you want to perform the Question-Answering on.

4. For each model, you can change the path of the ChartQA-Split dataset and the path where you want to save the results.

5. Some models (*GPT-4o and Gemini*) require an API key to run the model. You can insert your API key in the script to run the model.

6. To run any model, you can simply run the script '`<name_of_the_file>.py`' in the terminal or run the interactive notebook in the Jupyter notebooks provided.

### **Pipelining models through Gemini 1.5 Flash**
 
#### As mentioned in the paper, models like InternLM and CogAgent were experimentally found to not produce results in the required format. Hence, we have provided scripts to pipeline the results of these models to the Gemini model.

1. Run the model scripts independently with given chart type and question type. This will save the results in its original form.

2. Run `<model_name>_gemini.py` script to pipeline the results to the Gemini model.

3. The results for each category can be printed in the console or saved in a file.

### **Pipeline for GPT-4o**

#### We used a batched inference pipeline for GPT-4o model. Below are the steps to run the pipeline:

1. Run `GPT_batch_creation.py` to create the batched data for the GPT-4o model from the selected question type and chart type. This saves the batches in a JSONL file. 
*Note: Some batches can tend to be very large, and hence it is better to break them into smaller batches for faster processing.*

2. Run `GPT_batch_running.py` to upload the batches to the model. This will submit all batches and store the IDs of the submitted batches in a JSON file.

3. Run `GPT_batch_saving.py` to save the results of the submitted batches if they are completely processed. This will save the results in a JSONL file.

4. Run `GPT_results.py` to print the results of the GPT-4o model, loading the files from the saved responses.