# PolypClassification

CNN based binary classification of colorectal polyps using PyTorch.  


---

## Project Structure

```
IMA_project/PolypClassification/
├── preprocessing.py          # Preprocessing scripts 
├── train.py                 # Main training script
├── test.py                  # Validation script
├── new_test.py              # Customrized test script
├── model.py                 # Model definition
├── augmentation.py          # Data Augmentation, bbox related function aborted
├── dataset.py               # Dataset loading & transformation
├── utils.py                 # Helpers: checkpoint, metrics, etc.
├── train_config.py          # Training configuration
├── DATA_FILTERING.ipynb                  # Aborted Data filter script
├── DATA_FILTERING_FINAL.ipynb            # Aborted Data filter script with added heatmap and histograph
├── data_process.ipynb       # Main file for data processing used in this project
├── combining_augmented_polar.py          # combine augmentated dataset with original one
├── analyze_multi_misclassified.ipynb     # visualize common errors in discussion part
├── splits/                  # Save preprocessing split results
└── samples/                 # Logs and model checkpoints
```

---


### 0. Preprocess Data

Please follow `data_process.ipynb`

### 1. Install Dependencies

```bash
pip3 install -r requirements.txt --user
```



---

### 1.5(Optional). Data Augmentation

```bash
python3 augmentation.py
```

This will:
- read filtered polar csv file  `CSV_FILE` from `DATASET_ROOT` 
- generate a new csv file `OUTPUT_CSV ` and save augmentated data in `OUTPUT_DIR`

### 2. Preprocess Data

```bash
python3 preprocessing.py
```

This will:
- Read filtered POLAR dataset in `csv_path` and `data_dir`
- Split them into training/validation/test sets
- Save `(file_path, groud_truth_label)` outputs to `.split`
- Generate splitted dataset mean/std info for further training config
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

- `samples/logs/<exp_name>/` for tensorboard logs
- `samples/<exp_name>/` for each validation frequency model checkpoint
- `results/<exp_name>/` for best model across all epochs

---

### 5. Visualize Training Result

Use TensorBoard to monitor training:

```bash
tensorboard --logdir=samples/logs/
```

### 6. Test the Model
Modify `new_test.py` with parameters
- `checkpoint_path` the model checkpoint path
- `csv_file` to save all error paths for further error analysis in  `analyze_multi_misclassified.ipynb`

Then run 
```bash
python3 new_test.py
```

This will print the evaluation metrics on testset and the confusion matrix

### 7. Error Analysis

Please follow `analyze_multi_misclassified.ipynb` 