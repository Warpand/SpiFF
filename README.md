# SpiFF: Spatial Features Fingerprint

## Setup
Create a virtual environment or a conda environment and run:
```console
pip install -r requirements.txt
```
The code was developed with Python 3.10.12.

## Usage
To perform the default experiment, run:
```console
python3 -m spiff -E [WANDB_ENTITY] -P [WANDB_PROJECT]
```
If you don't want to use Wandb (for example for a debugging session), run:
```console
python3 -m spiff --no-wandb
```
### Configuration
The spiff/cfg.py file contains the default experiment configuration. You can change the
values there, or overwrite them at runtime with values from a JSON:
```console
python3 -m spiff -C [PATH_TO_JSON]
```
For example, using:
```json
{
  "learning_rate": 0.001,
  "model_config": {
    "gnn": "sage"
  }
}
```
will change the learning rate to 0.001 and the GNN type to GraphSAGE. Refer to the
docstrings in [spiff/cfg.py](spiff/cfg.py) for more information.

You can also overwrite some values using flags, for example:
```console
python3 -m spiff -E ... -P ... --batch-size 3072
```
If you mix flags and a configuration JSON, the flags take precedence.

To see all the possible flags, run:
```console
python3 -m spiff --help
```

### Dataset Filtering
For some molecules, rdkit might be unable to create an embedding or use a force field.
We cannot calculate 3D similarity measure for such molecules. To filter them out of a dataset,
run:
```console
python3 filter_data [PATH_TO_DATASET] [FILTERED_DATASET_SAVE_PATH]
```
The exemplary dataset in the repository is already filtered.

## Baselines
To perform baselines experiments, run:
```console
python3 -m baselines -E ... -P ... -s [DATASET] -t [TYPE]
```
Possible types are:
* frozen - pretrained SpiFF model is loaded from checkpoint, frozen and used as a feature extractor,
* tuned - pretrained SpiFF model is loaded and is fine-tuned for the specific task,
* trained - SpiFF model used as a feature extractor is trained for the specific task from zero,
* random - SpiFF model with random weights is frozen and used as a feature extractor.

Possible datasets are:
* QM9,
* BACE,
* HIV.

Just like before, you can used the -C flag to overload the configuration.
Run
```console
python3 -m baselines --help
```
to view all the flags. Pay attention to --checkpoint flag, that is necessary
for some of the baseline types.

### Notebooks
THe [notebooks](notebooks) directory contains Jupyter notebooks, that can be used to
run experiments using classical ML models and molecule fingerprints on the baseline
datasets.
