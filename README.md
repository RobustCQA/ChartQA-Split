# **ChartQA-Split**
## **Data for the ChartQA-Split dataset**
#### *This repository contains the dataset and implementation details for the ChartQA-Split dataset, as mentioned in the paper [Unraveling the Truth: Do LLMs really Understand Charts? A Deep Dive into Consistency and Robustness](https://arxiv.org/abs/2407.11229).*

### **Overview**
The ChartQA-Split dataset is a collection of question-answer pairs for charts, which is originally based on the ChartQA dataset. While the ChartQA dataset does not provide the complexity of the questions and charts, the ChartQA-Split dataset provides a split of the questions and charts based on their complexity. The dataset is split into eight different categories based on :
1. The complexity of the chart (*Simple* or *Complex*)
2. The complexity of the question (*Simple* or *Complex*)
3. The generation type of the question (*Human* or *Augmented*)

This fine grained split can help in understanding and improving the performance of models on different types of questions and charts. The dataset is provided in the `dataset` sub-directory of this repository. The `scripts` sub-directory contains the scripts used to inference various models on the dataset and replicate the results mentioned in the paper.

### **Strcuture**

This directory has been broken down into two sub-directories:
```
└── ChartQA-Split
    ├───dataset
    │   └───README.md
    ├───scripts
    │   └───README.md
    └───requirements.txt
```
***Details about each sub-directory are mentioned in the respective README files of each sub directory.***

### **Usage**
Clone the repository and follow the instructions in the README files of each sub-directory to use the dataset and scripts provided in this repository. 
Create a virtual environment and install the required dependencies using the following command:
```bash
conda create --name <env> --file requirements.txt
```



### **Citation**

To cite our work, please use the following BibTeX entry:
```bibtex
@misc{mukhopadhyay2024unravelingtruthllmsreally,
      title={Unraveling the Truth: Do LLMs really Understand Charts? A Deep Dive into Consistency and Robustness}, 
      author={Srija Mukhopadhyay and Adnan Qidwai and Aparna Garimella and Pritika Ramu and Vivek Gupta and Dan Roth},
      year={2024},
      eprint={2407.11229},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.11229}, 
}
```
