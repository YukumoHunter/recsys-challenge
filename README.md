# EB RecSys Challenge

## Results

### Baseline comparison

|     Method     |    AUC ↑     |    MRR ↑     |   nDCG@5 ↑   |  nDCG@10 ↑   |
| :------------: | :----------: | :----------: | :----------: | :----------: |
| Random (small) |    0.4998    |    0.3156    |    0.3489    |    0.4338    |
|  NRMS (small)  |    0.5299    |    0.3243    |    0.3625    |    0.4420    |
|  GERL (small)  | **_0.5820_** | **_0.3587_** | **_0.4032_** | **_0.4775_** |

### Neighbour sampling comparison

|             Method              |    AUC ↑     |    MRR ↑     |   nDCG@5 ↑   |  nDCG@10 ↑   |
| :-----------------------------: | :----------: | :----------: | :----------: | :----------: |
|           GERL (demo)           | **_0.5344_** | **_0.3231_** | **_0.3610_** | **_0.4436_** |
| GERL (demo) + weighted sampling |    0.5258    |    0.3172    |    0.3544    |    0.4377    |

## Setup

Create a virtual environment and install the needed dependencies:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.lock
```

Or alternatively, use [Rye](https://rye.astral.sh/):

```sh
rye sync
. .venv/bin/activate
```

From the root of the repository, run the preprocessing script:

```sh
python -m src.recsys_challenge.dataset.preprocess.auto
```

To train the model, run:

```sh
python -m src.recsys_challenge.training
```

The default options for training are set to the params we used for training.
Options for the training script can be printed using:

```sh
python -m src.recsys_challenge.training --help
```
