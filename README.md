# Code for More power multiple testing under dependence via randomization


## Setup

With `python==3.10` installed, run `pip install -r requirements.txt`.

## Reproducing figures

From the repo root directory, run:
```
EXP=<experiment name>; python src/main.py --processes <# of processors to use> --exp ${EXP} --out_dir results/${EXP} --result_dir figures/${EXP} --no_save_out
```

where `<experiment name>` can be replaced with 
- `gaussian_rho` to reproduce results relating to e-BH and randomized variants of e-BH
- `gaussian_p_rho` to reproduce figures relating to BY, U-BY
- `guo_rao` to reproduce results where the BY procedure is sharp
