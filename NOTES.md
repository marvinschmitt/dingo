## Dev setup for FMPE branch

```
conda create --name dingo python=3.9
conda install -c conda-forge dingo-gw=0.5.4
git clone https://github.com/dingo-gw/dingo.git
cd dingo
git checkout FMPE
```

## Fixes on the branch

- In `dingo.gw.gwutils`: `from scipy.signal import tukey` -> `from scipy.signal.windows import tukey`
