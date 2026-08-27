# Bridging the Gap Between Wearable Sensors and Clinical Practice in Agoraphobia and Panic Disorder: Single-Case Formative Evaluation of a Multi-Stage Large Language Model Pipeline

> *Ugonna Oleh, Alla Machulska, Roman Obermaisser, and Tim Klucken*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![R 4.2+](https://img.shields.io/badge/R-4.2%2B-blue.svg)](https://www.r-project.org/)
[![Model: Gemma-3-27B-it](https://img.shields.io/badge/LLM-Gemma--3--27B--it-orange.svg)](https://huggingface.co/google/gemma-3-27b-it)

This repository contains the complete software codebase, database schemas, prompt configurations, synthetic validation datasets, and statistical analysis scripts for the multi-stage, decoupled Large Language Model (LLM) pipeline presented in our manuscript. 

To resolve the core challenges of clinical safety and reproducibility, the system separates deterministic mathematical computations from generative language synthesis, establishing strict architectural guardrails against numerical hallucinations.

---

## 🏗️ System Architecture

The pipeline is fully decoupled into five specialized, sequential stages. Unlike monolithic LLM applications, this modular structure ensures that clinical biometrics are processed using deterministic mathematical libraries before any text is generated.

```
                    +--------------------------------+
                    |      Commercial Smartwatch     |
                    |   (Garmin IMU, Heart Rate)     |
                    +----------------+---------------+
                                     |
                                     v
+------------------+        +--------+-------+        +------------------+
|  Daily Patient   |        |  Local Loader  |        |    Clinician-    |
|   Digital Logs   |        |    Module      |        |   Injected CBT   |
+--------+---------+        +--------+-------+        |    Literature    |
         |                           |                +--------+---------+
         |                           v                         |
         |                  +--------+-------+                 |
         |                  |  Deterministic |                 |
         |                  |  Analyzer (Math|                 |
         |                  +--------+-------+                 |
         |                           |                         |
         |                           v                         v
         |                  +--------+-------+        +--------+---------+
         +----------------->|    Central     |<-------|  Local Neo4j Graph|
                            |  Orchestrator  |        |    Database      |
                            +--------+-------+        +------------------+
                                     |
                                     v
                            +--------+-------+
                            |  Multi-Persona |
                            | Reporter (LLM) |
                            +--------+-------+
                                     |
             +-----------------------+-----------------------+
             |                       |                       |
             v                       v                       v
     [ Persona 1 ]           [ Persona 2 ]           [ Persona 3 ]
    Clinical Research       Therapist Summary       Patient-Facing
       Log (Long)             Report (Short)        Empathetic Log
```

1. **Local Loader Module (`code/local_loader.py`):** Handles secure, private ingestion of multimodal sensor streams, daily digital journals, and initial clinical profiles, standardizing timestamps and aligning 7-day longitudinal weekly frames.
2. **Deterministic Analyzer Module (`code/deterministic_analyzer.py`):** Extracts physiological and behavioral features (e.g., 120-second physiological arousal bouts and 300-second psychomotor fidgeting bouts) from raw sensor streams using strict, scientifically calibrated mathematical thresholds.
3. **Central Orchestrator Module (`code/central_orchestrator.py`):** Synthesizes outputs, manages the longitudinal context, coordinates the recursive memory loop (passing the summary from Week $N-1$ to Week $N$), and manages ablation toggles.
4. **RAG-Retrieval Module (`code/RAG_retrieval.py`):** Queries a local Neo4j Graph Database and dense vector index to retrieve context-specific, validated Cognitive Behavioral Therapy (CBT) literature based on active patient triggers.
5. **Multi-Persona Reporter Module (`code/multi_persona_reporter.py`):** Coordinates quantized, local workstation inference (`google/gemma-3-27b-it` in `bfloat16`) to generate three customized narrative formats (Long-Form Research Log, Short-Form Psychologist Report, and Empathetic Patient Feedback) protected by strict epistemic hedging guardrails.

---

## 💻 Workstation & Hardware Specifications

All local text generation and data processing was executed offline on a secure, air-gapped workstation to guarantee absolute participant privacy.

* **Operating System:** Ubuntu 22.04 LTS (x86_64)
* **CPU:** AMD Threadripper PRO 5955WX (16 Cores, 32 Threads, 4.0 GHz Base / 4.5 GHz Boost)
* **GPU:** 1x NVIDIA RTX 6000 Ada Generation (48 GB GDDR6 VRAM with ECC)
* **RAM:** 128 GB DDR5 ECC Registered RAM
* **Storage:** 2 TB NVMe PCIe Gen4 SSD
* **Local LLM Execution Environment:** Quantized local inference utilizing the `bfloat16` precision pipeline of the **Gemma-3-27b-it** model. Gradients and backpropagation were disabled during execution, utilizing the GPU's wide dynamic range strictly for activation and Key-Value (KV) cache stability over extended longitudinal prompt context lengths.

---

## 🤖 Local LLM (Gemma-3) Setup & Authentication

Our multi-persona reporting pipeline utilizes Google's open-weights **Gemma-3-27b-it** model running locally on the workstation. Because Gemma models are gated on Hugging Face, follow these setup steps prior to running inference:

### 1. Request Model Access
1. Visit the [Gemma-3-27b-it Hugging Face Page](https://huggingface.co/google/gemma-3-27b-it).
2. Log into your Hugging Face account and accept the Gemma Terms of Use.

### 2. Authenticate Your Workstation
Log in via the Hugging Face CLI so that your environment can load the model weights:
```bash
huggingface-cli login
```
Alternatively, set your Hugging Face access token as an environment variable:
```bash
# Linux / macOS
export HF_TOKEN="your_hf_access_token"

# Windows PowerShell
$env:HF_TOKEN="your_hf_access_token"
```

### 3. Precision, Quantization & Memory Configuration
* **Default Research Config (48 GB VRAM):** By default, `code/multi_persona_reporter.py` loads `google/gemma-3-27b-it` in `bfloat16` precision with 4-bit (`load_in_4bit=True` via `bitsandbytes`) quantization and device map auto-allocation. This consumes ~18–22 GB VRAM for weights, reserving the remaining memory for long-context KV caching across multi-week prompts.
* **Pre-Downloaded Local Weights:** If running on an offline/air-gapped machine with pre-downloaded weights, pass the local directory path:
  ```bash
  python code/central_orchestrator.py --model_path /local/weights/gemma-3-27b-it
  ```
* **Lower-Memory GPUs (16–24 GB VRAM):** If executing on consumer workstations, you can target smaller parameter configurations:
  ```bash
  python code/central_orchestrator.py --model_path google/gemma-3-12b-it
  # or: google/gemma-3-4b-it
  ```
* **Offline Emulation / CPU Testing Mode:** To verify pipeline execution without downloading LLM weights or requiring a CUDA GPU, pass `mock`:
  ```bash
  python code/central_orchestrator.py --model_path mock
  ```

---

## 🛠️ Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/wearable-mental-health/wearable-mental-health-pipeline.git
cd wearable-mental-health-pipeline
```

### 2. Configure Environment

#### Using Conda (Recommended):
```bash
conda env create -f environment.yml
conda activate mental-health-pipeline
```

#### Using pip:
```bash
pip install -r requirements.txt
```

### 3. Initialize Local Neo4j Graph Database
This system relies on a local instance of Neo4j to manage the CBT clinical recommendations and psychiatric ontologies.
1. Download and install [Neo4j Community Edition](https://neo4j.com/download-center/).
2. Start the Neo4j console service:
   ```bash
   neo4j start
   ```
3. Load the clinical ontology schemas, relationship rules, and CBT literature nodes:
   ```bash
   cypher-shell -u neo4j -p YourSecurePassword -f database/neo4j_schema.cypher
   ```

*(Note: The `code/RAG_retrieval.py` script also includes an embedded vector cache fallback mode for offline testing without a live Neo4j server).*

---

## 🚀 Running the Pipeline

To run the full multi-stage pipeline utilizing our synthetic, de-identified sample dataset:

```bash
python code/central_orchestrator.py \
  --input_sensor_data sample_data/deidentified_inputs.json \
  --previous_report sample_data/evaluated_reports/week_17_summary.json \
  --output_dir output/week_18/ \
  --model_path google/gemma-3-27b-it \
  --rag_enabled true \
  --memory_enabled true
```

### Running Comparative Baseline Controls & Ablations

To replicate the experimental control conditions described in the Results section:

* **Non-LLM Statistical Baseline Dashboard (`baseline_template.py`):**
  ```bash
  python code/baseline_template.py --input_sensor_data sample_data/deidentified_inputs.json --output_dir output/baseline/
  ```
* **"No-RAG" Ablation (Bypassing CBT literature retrieval and patient journals):**
  ```bash
  python code/central_orchestrator.py --input_sensor_data sample_data/deidentified_inputs.json --rag_enabled false
  ```
* **"No-Memory" Ablation (Disabling recursive historical summaries to evaluate temporal blindness):**
  ```bash
  python code/central_orchestrator.py --input_sensor_data sample_data/deidentified_inputs.json --memory_enabled false
  ```

---

## 📊 Statistical Evaluation Replications

To calculate the inter-rater agreement and descriptive central tendency metrics of the clinical expert study ($N=7$ independent evaluators; 42 total evaluation sheets across 3 landmark weeks):

1. Navigate to the evaluation directory:
   ```bash
   cd evaluation/
   ```
2. Execute the R statistical script to compute Gwet's second-order agreement coefficients ($AC_2$) with linear ordinal weights and corresponding 95% Confidence Intervals:
   ```bash
   Rscript gwet_ac2_analysis.R --input raw_expert_ratings.csv
   ```

---

## 📂 Repository File Manifest

```
wearable-mental-health-pipeline/
├── LICENSE                               # Open-source MIT License
├── README.md                             # Installation, system requirements, and quick-start guide
├── environment.yml                       # Conda environment configuration
├── requirements.txt                      # Exact pip dependency versions
├── code/                                 # Core modular pipeline source files
│   ├── local_loader.py                   # Secure ingestion and timeframe parsing
│   ├── deterministic_analyzer.py         # Mathematical feature extraction (120s / 300s thresholds)
│   ├── central_orchestrator.py           # Coordinating class and recursive context memory loop
│   ├── RAG_retrieval.py                  # Vector database queries and top-k retrieval mechanics
│   ├── multi_persona_reporter.py         # Quantized Gemma-3 local workstation inference config
│   └── baseline_template.py              # Non-LLM statistical baseline script used for comparison
├── prompts/                              # Exact prompt configurations and instruction sets
│   ├── system_instructions/              # Prompt configurations for the 3 generated report personas
│   │   ├── clinical_research_log.txt     # Persona 1: Long-Form Clinical Research Log
│   │   ├── psychologist_summary.txt      # Persona 2: Short-Form Psychologist Report
│   │   └── patient_narrative.txt         # Persona 3: Empathetic Patient Feedback Narrative
│   └── evaluation_judges/                # Prompts utilized for the LLM-as-a-judge evaluation
│       ├── faithfulness.txt              # Faithfulness & hallucination audit prompt
│       ├── answer_relevance.txt          # Clinical query relevance prompt
│       └── context_precision.txt         # Retrieved RAG context precision prompt
├── database/                             # Database architecture files
│   ├── neo4j_schema.cypher               # Graph database ontology schemas and CBT guidelines
│   └── clinical_literature_sources.md    # Source inventory of psychiatric textbooks & guidelines
├── evaluation/                           # Clinical expert evaluation instruments & data
│   ├── gwet_ac2_analysis.R               # R statistical script computing weighted AC2 and CIs
│   ├── raw_expert_ratings.csv            # Raw 1–5 scoring matrix of the seven experts (N=7, 42 reports)
│   └── clinical_questionnaire.pdf        # Standardization rubrics and questionnaire sheet template
└── sample_data/                          # Synthetic files to execute the pipeline locally
    ├── deidentified_inputs.json          # Anonymized mock physiological and behavioral data streams
    └── evaluated_reports/                # The exact six reports evaluated by clinical experts
        ├── week_01_long_form.txt         # Landmark Week 1 Long-Form Research Log
        ├── week_01_short_form.txt        # Landmark Week 1 Short-Form Psychologist Report
        ├── week_18_long_form.txt         # Landmark Week 18 Long-Form Research Log
        ├── week_18_short_form.txt        # Landmark Week 18 Short-Form Psychologist Report
        ├── week_23_long_form.txt         # Landmark Week 23 Long-Form Research Log
        ├── week_23_short_form.txt        # Landmark Week 23 Short-Form Psychologist Report
        └── week_17_summary.json          # Week 17 recursive memory injection artifact
```

---

## 🔒 Ethical Data Use & De-Identification Policy

Due to the rich, highly detailed longitudinal nature of our $N=1$ single-case study, the raw physiological sensor streams and daily qualitative text journals are permanently withheld to protect participant privacy. In an $N=1$ design, publishing raw, high-density longitudinal data creates a significant risk of deductive re-identification, violating our institutional review board (IRB) ethical agreements.

To facilitate secure code verification and pipeline testing, we have provided:
1. **`sample_data/deidentified_inputs.json`:** A structurally identical, fully synthetic patient dataset mapping mock physiological heart rate metrics, hand-fidgeting IMU bouts, and synthetic journal logs.
2. **`sample_data/evaluated_reports/`:** A set of anonymized clinical reports generated during the study's landmark evaluation weeks (Weeks 1, 18, and 23) that were scored by our independent clinical experts.

---

## 📖 Citation

If you utilize this architecture, prompts, or evaluation methodology in your research, please cite our manuscript:

```bibtex
@article{oleh2026bridging,
  title={Bridging the Gap Between Wearable Sensors and Clinical Practice in Agoraphobia and Panic Disorder: Single-Case Formative Evaluation of a Multi-Stage Large Language Model Pipeline},
  author={Oleh, Ugonna and Machulska, Alla and Obermaisser, Roman and Klucken, Tim},
  journal={JMIR Mental Health / Journal of Medical Internet Research},
  year={2026},
  publisher={JMIR Publications}
}
```

---

## 📄 License

This codebase is open-source and licensed under the **MIT License**. See the `LICENSE` file for details.
