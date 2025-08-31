# Bongard Problem Image Generator

This repository implements a **text-to-image transformer** for generating Bongard Problem pairs.  
Experiments were conducted with **Python 3.9.13**. All library versions are pinned in [`requirements.txt`](./requirements.txt) for reproducibility.

---

## Configuration

Model and dataset configuration is centralised in [`config.py`](./config.py).  
This file controls:

- **Dataset selection** (`symbolic`, `english`, `minimal`)  
- **Training hyper-parameters** (optimiser, learning rate, epochs, feature weights, etc.)  

To change dataset or modify hyper-parameters, edit the fields in `config.py` before running training.

---

## Data Management

Datasets are stored under [`data/`](./data/) with the following structure:

```
data/
  ├── english/
  │   ├── english_words_image_relationships.csv
  │   └── expanded_english_words_image_relationships.csv
  ├── symbolic/
  │   ├── symbolic_words_image_relationships.csv
  │   └── expanded_symbolic_words_image_relationships.csv
  └── minimal/
      ├── minimal_words_image_relationships.csv
      └── expanded_minimal_words_image_relationships.csv
```

- **`{dataset}_words_image_relationships.csv`**: contains the base rows in the format:
  ```
  "input text",BP#
  ```
  Example:
  ```
  "Big vs small",2
  ```

- **`expanded_{dataset}_words_image_relationships.csv`**: generated automatically; contains **36 contrastive pairs** per Bongard Problem row.

### Expanding Data
Whenever you add or modify rows in `{dataset}_words_image_relationships.csv`, regenerate the expanded dataset:

```bash
cd executables
python expand_data.py
```

This script expands each row into a full set of contrastive pairs and writes them into `expanded_{dataset}_words_image_relationships.csv`.

---

## Training

To train a model:

```bash
cd executables
python train.py
```

- Checkpoints, training loss logs, and debug image previews will be written to the output directory call training_debug.  
- Training defaults to **100 epochs**, but can be stopped early if loss plateaus.  

---

## Generating Images

Once training completes:

```bash
python generate_images.py
```

- Uses the dataset’s `bongard_problems_to_test.csv` as the source of test prompts.  
- Outputs generate images to `/model_answers_{dataset}_sentences`.  
- You may freely edit this CSV to add or remove prompts for evaluation.  

---