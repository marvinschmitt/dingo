## Dev setup for FMPE branch

We want to install the FMPE branch of Dingo, but pip install doesn't work because the wheel fails building. We can use conda to install the dependencies at the stage of the FMPE branch (v0.5.4), then uninstall Dingo again so that we
can use the locally developed version.

First: create a new conda environment and install the FMPE version of Dingo, then remove:

```
conda create --name dingo python=3.9
conda install -n base conda-forge::mamba
mamba install -c conda-forge dingo-gw=0.5.4
mamba install -c conda-forge torchdiffeq wandb
pip install chainconsumer
conda remove --force dingo-gw
```

Then clone the FMPE branch of the repository and install it in editable mode without dependencies:

```
git clone https://github.com/dingo-gw/dingo.git
cd dingo
git checkout FMPE
pip install -e . --no-deps
```

## Fixes on the branch

- In `dingo.gw.gwutils`: `from scipy.signal import tukey` -> `from scipy.signal.windows import tukey`

Add CLI to generate ASD dataset:

```
# in dingo/gw/noise/generate_dataset.py
# ...

def main():
    generate_dataset()

if __name__ == "__main__":
    main()
```

## Running the code

Run these commands from the base repo directory to generate the datasets:

```
python -m dingo.gw.dataset.generate_dataset --settings dev_scripts/waveform_dataset_settings.yaml --out_file dev_scripts/training_data/waveform_dataset.hdf5

python -m dingo.gw.noise.generate_dataset --settings_file dev_scripts/asd_dataset_settings.yaml --data_dir dev_scripts/training_data/asd_dataset
```
