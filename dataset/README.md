# **Data for ChartQA-Split dataset**

## **Below mentioned are the files for running the Question-Answering on ChartQA-Split**


```
└─ dataset
  ├───complex
  │   ├───png
  │   │   └───<QA_ID>.png
  │   ├───test_augmented_complex.json
  │   ├───test_augmented_simple.json
  │   ├───test_human_complex.json
  │   └───test_human_simple.json
  │   
  └───simple
      ├───png
      │   └───<QA_ID>.png
      ├───test_augmented_complex.json
      ├───test_augmented_simple.json
      ├───test_human_complex.json
      └───test_human_simple.json
```

### **General Information about the files:**

1. The naming scheme for each folder is as follows: `<chart_type>` where chart_type represents the complexity of the chart in the question-answer pairs. For example, `complex` contains the question-answer pairs which correspond to complex charts.

2. In every folder, 4 JSON files are present. Each of the JSONs contains information about the QA pairs for the respective chart type. The naming scheme for each JSON file is as follows: `test_<generation_type>_<question_type>.json` where generation_type represents the generation type of the question: *augmented* or *human* and question_type represents the complexity of the question: *simple* or *complex*.

3. The `png` folder contains the images for the respective QA pairs. The naming scheme for each image is as follows: `<QA_ID>.png` where QA_ID represents the unique identifier for the question-answer pair.

### **Description of the JSON files:**

```
    {
        "imgname": '<QA_ID>.png',
        "query": '<question>',
        "label": '<gold_answer>'
    }
```