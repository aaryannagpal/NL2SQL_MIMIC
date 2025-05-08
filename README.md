# NL2SQL for MIMIC-IV Database

A research project implementing a two-stage pipeline for accurate Natural Language to SQL (NL2SQL) conversion specialized for the MIMIC-IV medical database.

## Project Overview

This project develops and evaluates an advanced approach to convert natural language questions into SQL queries for medical database interactions. The core contribution is a two-stage architecture consisting of:

1. **Generation Model (M1)**: Converts natural language questions into SQL queries
2. **Validation Model (M2)**: Identifies and corrects errors in generated queries

The implementation explores progressive prompting strategies and fine-tuning techniques to optimize performance while working within computational constraints:

- Zero-shot, few-shot, and schema-aware prompting approaches
- Parameter-efficient fine-tuning with LoRA
- Comprehensive query evaluation framework
- Specialized handling for medical database complexities

## Key Features

- **Progressive Model Development**: Systematic evaluation of nine LLMs across three prompting strategies
- **Multi-dimensional Evaluation**: Beyond execution results, incorporating structural and component-level metrics
- **MIMIC-IV Specialization**: Tailored approach for complex medical database schemas
- **Empty Results Handling**: Robust evaluation despite predominant empty result sets
- **Memory-Efficient Implementation**: Optimized for execution within 24GB GPU constraints

## Current Development Status

- [x] MIMIC-IV database configuration and adaptation
- [x] Comprehensive query evaluation framework
- [x] Generation model (M1) zero-shot implementation
- [x] Generation model (M1) few-shot refinement
- [x] Generation model (M1) schema-aware enhancement
- [x] Generation model (M1) fine-tuning
- [x] Validation model (M2) zero-shot implementation
- [x] Validation model (M2) few-shot refinement
- [x] Validation model (M2) schema-aware enhancement
- [ ] Validation model (M2) fine-tuning
- [ ] Pipeline integration and optimization
- [ ] End-to-end evaluation and benchmarking
- [ ] Final thesis documentation

## Technologies

- **Base Models**: Phi-4, Qwen 1.5/2.5, MedQwen, Meditron 7B, Medalpaca 13B, DuckDB NSQL 7B, SQLCoder 7B
- **Fine-Tuning**: LoRA with 4-bit quantization
- **Database**: MIMIC-IV (adapted from EHRSQL 2024 competition schema)
- **Evaluation**: Custom metrics for table/column access accuracy, component similarity, execution plan analysis

## Getting Started

### Prerequisites
- NVIDIA GPU with 24GB+ VRAM
- Python 3.9+
- SQLite database with modified MIMIC-IV schema
- EHRSQL 2024 dataset

### Installation
[To be added]

### Usage
[To be added]

## License

[To be added]