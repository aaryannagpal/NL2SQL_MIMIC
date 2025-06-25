# NL2SQL for MIMIC-IV Database

This repository accompanies the thesis project: Natural Language to SQL (NL2SQL) for the MIMIC-IV Medical Database

## Motivation

The complexity of medical databases like MIMIC-IV often prevents clinicians and researchers from fully utilizing valuable healthcare data. By enabling natural language queries over structured clinical data, this project aims to lower technical barriers and support more inclusive, data-driven decision-making in medicine.

## Objectives

### Comparative Model Analysis for MIMIC Query Tasks
- Conduct a comprehensive analysis of different language models on MIMIC-IV natural language queries to assess their performance across various implementation techniques:
  1. Evaluate both general-purpose and medical-domain LLMs (including Phi-4, Qwen 1.5/2.5, MedQwen, Meditron, Medalpaca, and specialized SQL models)
  2. Compare performance under zero-shot and few-shot prompting regimes
  3. Assess model capabilities across two distinct tasks:
     - Query generation: Converting natural language questions to executable SQL
     - Query validation: Verifying and correcting generated SQL for schema compliance and logical consistency
  4. Develop robust evaluation metrics focused on execution success, result accuracy, and clinical relevance

### Optimized Two-Stage Pipeline Development
- Building on the model analysis findings, the primary objective is to develop and implement an efficient two-stage natural language to SQL pipeline specifically optimized for the MIMIC-IV database:
  1. Select the best-performing models for both generation and validation stages based on comparative analysis
  2. Fine-tune selected models to enhance performance on medical text-to-SQL tasks
  3. Design an integrated pipeline architecture that balances accuracy with practical efficiency
  4. Implement schema-aware validation mechanisms to ensure query correctness
  5. Evaluate the end-to-end pipeline against baseline approaches using clinically relevant metrics

## Methodology

This work adopts a rigorous, multi-phase methodology:

1. **Dataset Preparation**: The MIMIC-IV database and EHRSQL 2024 benchmark are adapted for NL2SQL evaluation, ensuring schema alignment and clinical relevance.
2. **Model Selection**: Both general-purpose and medical-domain LLMs are selected, including Phi-4, Qwen, MedQwen, Meditron, Medalpaca, and SQL-specialized models.
3. **Prompting Strategies**: Each model is evaluated under zero-shot, few-shot, and schema-aware prompting to assess adaptability and performance.
4. **Two-Stage Pipeline Design**:
   - **Stage 1 (Generation)**: The best-performing models generate SQL from natural language queries.
   - **Stage 2 (Validation)**: A separate model validates and corrects generated SQL, focusing on schema compliance and logical consistency.
5. **Fine-Tuning**: LoRA-based parameter-efficient fine-tuning is applied to selected models to improve domain-specific performance.
6. **Evaluation**: A comprehensive framework is used, including execution accuracy, structural/component metrics, and error categorization. Clinical relevance is prioritized in metric design.
7. **Analysis & Visualization**: Error patterns and model behaviors are analyzed using custom scripts and visualizations to inform future improvements.

## Contributions

- A memory-efficient NL2SQL pipeline for the MIMIC-IV database, with open-source code for reproducibility.
- Systematic benchmarking of multiple LLMs and prompting strategies in the medical SQL generation context.
- A robust evaluation framework and detailed error analysis tailored to the needs of clinical data retrieval.
- Insights and recommendations for deploying NL2SQL systems in real-world healthcare research settings.

## Directory Structure

- `src/` - Main entry point and orchestration scripts
- `model/` - Model implementations for M1 and M2 (few-shot, finetune, schema-aware, zeroshot)
- `analysis/` - Data and model analysis scripts, visualizations, and evaluation outputs
- `data/` - Datasets, MIMIC-IV database files, and schema information
- `helper_scripts/` - Utility scripts for data sampling, cleaning, and scoring
- `utils/` - Core utility modules for dataset handling, query analysis, evaluation, and visualization
- `results/` - Output results from experiments and evaluations

## Technologies

- **Base Models**: Phi-4, Qwen 1.5/2.5, MedQwen, Meditron 7B, Medalpaca 13B, DuckDB NSQL 7B, SQLCoder 7B
- **Fine-Tuning**: LoRA with 4-bit quantization
- **Database**: MIMIC-IV (adapted from EHRSQL 2024 competition schema)
- **Evaluation**: Custom metrics for table/column access accuracy, component similarity, execution plan analysis
- **Visualization**: Matplotlib, Seaborn for analysis plots

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd NL2SQL_MIMIC
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have access to the MIMIC-IV database and EHRSQL 2024 dataset (see `data/` folder).

## Usage

- Main pipeline entry point:
  ```bash
  python src/main.py --config config.py
  ```
- Helper scripts for data sampling, cleaning, and evaluation are in `helper_scripts/`.
- Model-specific scripts (few-shot, finetune, schema-aware, zeroshot) are in `model/m1/` and `model/m2/`.
- Analysis and visualization scripts are in `analysis/`.

## Example Directory Structure

```
NL2SQL_MIMIC/
├── src/
├── model/
├── analysis/
├── data/
├── helper_scripts/
├── utils/
├── results/
├── requirements.txt
├── config.py
└── README.md
```

## References

1. Michael Han, Daniel Han, and Unsloth team. Unsloth, 2023. URL http://github.com/unslothai/unsloth.
2. Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-Wei H Lehman, Mengling Feng, Marzyeh Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iv: A freely accessible critical care database. Scientific Data, 10(1):1–14, 2023. doi:10.1038/s41597-023-02055-7.
3. Ayush Kumar, Parth Nagarkar, Prabhav Nalhe, and Sanjeev Vijayakumar. Deep learning driven natural languages text to sql query conversion: A survey. arXiv preprint arXiv:2208.04415, 2022. URL https://arxiv.org/abs/2208.04415.
4. Gyubok Lee, Sunjun Kweon, Seongsu Bae, and Edward Choi. Overview of the EHRSQL 2024 shared task on reliable text-to-SQL modeling on electronic health records. In Proceedings of the 6th Clinical Natural Language Processing Workshop, pages 644–654, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi:10.18653/v1/2024.clinicalnlp-1.62. URL https://aclanthology.org/2024.clinicalnlp-1.62/.
5. Ali Mohammadjafari, Anthony S. Maida, and Raju Gottumukkala. From natural language to sql: Review of llm-based text-to-sql systems, 2025. URL https://arxiv.org/abs/2410.01066.
6. Zheng Ning, Yuan Tian, Zheng Zhang, Tianyi Zhang, and Toby Jia-Jun Li. Insights into natural language database query errors: From attention misalignment to user handling strategies. arXiv preprint arXiv:2402.07304, 2024. URL https://arxiv.org/abs/2402.07304.
7. Richard Tarbell, Kim-Kwang Raymond Choo, Glenn Dietrich, and Anthony Rios. Towards understanding the generalization of medical text-to-sql models and datasets. arXiv preprint arXiv:2303.12898, 2023. URL https://arxiv.org/abs/2303.12898.
8. Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task, 2018. URL https://aclanthology.org/D18-1425/.
9. Xiaohu Zhu, Qian Li, Lizhen Cui, and Yongkang Liu. Large language model enhanced text-to-sql generation: A survey. arXiv preprint arXiv:2410.06011, 2024. URL https://arxiv.org/abs/2410.06011.
10. Angelo Ziletti and Leonardo D’Ambrosi. Retrieval augmented text-to-sql generation for epidemiological question answering using electronic health records. In Proceedings of the 6th Clinical Natural Language Processing Workshop, pages 47–53. Association for Computational Linguistics, 2024. doi:10.18653/v1/2024.clinicalnlp-1.4. URL http://dx.doi.org/10.18653/v1/2024.clinicalnlp-1.4.

## License

This project is licensed under the MIT License. See the LICENSE file for details.