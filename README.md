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
docstrings in spiff/cfg.py for more information.

You can also overwrite some values using flags, for example:
```console
python3 -m spiff -E ... -P ... --batch-size 3072
```
If you mix flags and a configuration JSON, the flags take precedence.

To see all the possible flags, run:
```console
python3 -m spiff --help
```
