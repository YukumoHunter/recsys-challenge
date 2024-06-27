# EB RecSys Challenge

Place the downloaded dataset files in the `data/download` folder and run the extraction script:

```bash
TODO
```

Create a virtual environment and install the needed dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.lock
```
Or alternatively, use [Rye](https://rye.astral.sh/):
```bash
rye sync
. .venv/bin/activate
```

From the root of the repository, run the preprocessing script:
```bash
python -m src.recsys_challenge.dataset.preprocess.auto
```

To train the model, run:
```bash
TODO
```