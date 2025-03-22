# PolypClassification

CNN based binary classification of colorectal polyps using PyTorch.  


---

## Project Structure

```
IMA_project/PolypClassification/
├── preprocessing.py          # Preprocessing scripts 
├── train.py                 # Main training script
├── test.py                  # Validation script
├── model.py                 # Model definition
├── dataset.py               # Dataset loading & transformation
├── utils.py                 # Helpers: checkpoint, metrics, etc.
├── train_config.py          # Training configuration
└── samples/                 # Logs and model checkpoints
```

---


### 1. Install Dependencies

```bash
pip3 install -r requirements.txt --user
```



---

### 2. Preprocess Data

```bash
python3 preprocessing.py
```

This will:
- Read filtered POLAR dataset in `csv_path` and `data_dir`
- Split them into training/validation/test sets
- Save `(file_path, groud_truth_label)` outputs to `.split`

output example:
```
/Polar_Dataset/validation_set/507983c0f9d145e7a4c48f5cb0e8e08e/6aa4b2e5-e890-428c-a9e5-713a3c890dce.png	adenoma
```

---

### 3. Configure Training

Edit `train_config.py` to set your experiment parameters:

```python
device = "cuda:0"  # or "cuda:3"
exp_name = "baseline_experiment"

train_split_dir = "datasets/train"
val_split_dir = "datasets/val"
batch_size = 32
epochs = 50
val_freq = 1
...
```

---

### 4. Train the Model

```bash
python3 train.py
```

Logs and checkpoints will be saved to:

- `samples/logs/<exp_name>/`
- `samples/<exp_name>/`
- `results/<exp_name>/`

---

### 5. Visualize Training

Use TensorBoard to monitor training:

```bash
tensorboard --logdir=samples/logs/
```


