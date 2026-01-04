## Distribution, Dosage, Density: What Dominates a BERT-like Model’s Learning to Disambiguate Word Senses?

This repository contains the resources used in our study, including:

- experimental results for all models and datasets
- manually annotated data used in the analysis
- training and evaluation code

The repository is provided **for anonymous review only**.

------

## 📁 Repository Structure

```css
project-root/
│
├── data/
│   ├── semcor
│   ├── semcor_5
│   ├── omsti_40
│   ├── omsti_80
│   └── annotations/
|	|	├──annotated_data_stage1
|	|	├──annotated_data_stage1_analysis
|	|	└── semcor-key
│
├── annotation interface/
│   ├── index.html
├── results/
│   ├── bert/
│   └── roberta/
│
├── training.py
|
├── predictor.py
|
├── __init__.py
│
├── requirement.txt
│
└── README.md

```

------

## 🧠 Research Overview

This work presents a probing study of BERT-like models for Word Sense Disambiguation (WSD), examining the influence of three factors:

| Factor           | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| **Distribution** | Sense-level performance under different data distributions   |
| **Dosage**       | Effect of unique training exposure on internal layer behavior |
| **Density**      | Influence of contextual semantic cue density on attention    |

The repository includes code and artifacts supporting all reported analyses.

------

## ⚙️ Environment & Dependencies

```
Python == 3.10.0
PyTorch == 2.6.0
Transformers == 4.53.0
numpy == 2.2.4
tqdm == 4.67.1
matplotlib == 3.10.1
scikit-learn == 1.6.1
joblib == 1.4.2
```

Install:

```
pip install -r requirements.txt
```

------

## ▶️ Training

Scripts for supervised fine-tuning BERT-like models are located in:

```bash
python main.py --task train \
  --train_data path/to/train.json \
  --artifacts outputs/run1 \
  --orig_data SemCor
```

## 🔁 Reproducing Results

#### WSD Evaluation

```bash
python main.py --task eval \
  --model path/to/model \
  --test_data path/to/test.json \
  --train_data path/to/train.json
```

#### Compute Probabilities per Layer

```bash
python main.py --task prob \
  --model path/to/model \
  --test_data path/to/test.json \
  --train_data path/to/train.json
```

#### Compute WHR

```
python main.py --task whr \
  --model path/to/model
```

## 📝 Annotation Data

This repository includes:

- Stage-1 annotation
   sense correctness
   perceived disambiguation difficulty
   top-ranked contextual keywords
- Stage-2 annotation
   keyword sets for density manipulation

Annotation files are stored in:

```
data/annotations/
```

annotation details are provided in:

```
annotation_interface/
```

## 📊 Result Files

Processed results used in the paper are provided under:

```
results/
```

Including:

- Mean probability per layer
- m-F1 / M-F1 scores on used test sets


