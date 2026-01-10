# AmonUsDetector
I am implementing a among us detector based on AmongUs Sideman Series


```
pip install uv
uv sync
conda create -n among_us python=3.11
conda activate among_us

```
To train model run:
```
uv run python -m src.data_module.generate generate_data --dest_folder=data/test_gen
```