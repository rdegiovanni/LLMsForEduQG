# LLMsForEduQG

This repo contains the replication package of our paper **Towards Reliable LLM-based Exam Generation. Lessons Learned and Open Challenges in an Industrial Project** 
accepted and the *Industry Showcase Track* of **ASE 2025**.

## Requirements
- Python 3
- an Open AI Key to run the GPT models.
- a Hugging Face API Key to run Llama, Mistral and DeepSeek models. You can adapt the code if these are locally deployed.


## Installation
You can install and run the experiments with the following commands:

1. `git clone https://github.com/rdegiovanni/LLMsForEduQG.git`

2. `cd LLMsForEduQG`

3. `pip install virtualenv`

4. `virtualenv venv`

5. `source venv/bin/activate`

6. `pip install -r requirements.txt`

7. `export OPENAI_API_KEY=...`

8. `export API_KEY_HUGGINGFACE=...`

9.
```
./LLMsForEduQG.sh -h

usage: LLMsForEduQG [-h] [-i INPUT] [-d DIR] [-N MAX] [-r RANDOM] [-m MODELS] [-sw PREFIX] [-p PROMPTS] [-ll] [-pl]
Testing framework to assess the performance of LLMs in generating multiple-choice education questions.
options:
-h, --help 								show this help message and exit
-i INPUT, --input INPUT 				Input .csv file with the ground truth questions.
-d DIR, --dir DIR, -o DIR, --out DIR	Output directory.
-N MAX, -n MAX, --max MAX				Maximum number of questions analysed from the dataset.
-r RANDOM, --random RANDOM				Random order of analysed questions.
-m MODELS, --models MODELS				Select the LLMs to run.
-sw PREFIX, --starts-with PREFIX		Selects all LLMs starting with the <prefix>.
-p PROMPTS, --prompts PROMPTS			Select the prompts to use.
-ll, --llms, --llm-list					List the supported LLMs.
-pl, --prompt-list						List the available prompts.

```

10. `./LLMsForEduQG.sh -i datasets/SciQ_test.csv -o results/Llama3170Instruct -m "Llama3170Instruct"`



## Results
You can download our empirical results from the experiments folder: [LLMs4EduQG_results.zip](https://github.com/rdegiovanni/LLMsForEduQG/blob/main/experiments/LLMs4EduQG_results.zip).

In the zip file, you will find two folders with the generated questions and analysis for the two datasets used in our experiments, namely, *SciQ* and the *Canterbury Question Bank* datasets. For each dataset, we provide the following information:

* `generated_questions.csv`: raw data extracted from all LLM executions.

* `clean_generated_questions.csv`: clean data only from the successful executions.

* `summary.csv`: for each LLM, the average values for all the effectiveness metrics.

* `statistics.csv`: results of the statistical tests between the different prompts.

* `metric.pdf`: for each metric, we provide a boxplot to ease their analysis and comparison.

* `choices_quality_analysis.csv`: analysis of the exact matches w.r.t. the ground truth for each generated question.

* `choices_summary.csv`: summarises the exact matches results for each LLM and prompt.


## Cite this paper
If you use our tool or results in your research, please cite our paper:

	@inproceedings{DegiovanniCabot2025,
	  author    = {Renzo Degiovanni and
		       Jordi Cabot},
	  title     = {Towards Reliable LLM-based Exam Generation. Lessons Learned and Open Challenges in an Industrial Project,
	  booktitle = {Industry Showcase Track of the 40th IEEE/ACM International Conference on Automated Software Engineering (ASE), Seoul, South-Korea, 2025},
	  publisher = {{IEEE}},
	  year      = {2025},
	}


