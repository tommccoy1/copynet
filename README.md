# CopyNet

This repository contains code for the CopyNet experiments in the paper (LINK TO PAPER ONCE AVAILABLE) by AUTHORS. This code is based on code from the [TP-Transformer repo](https://github.com/ischlag/TP-Transformer).

## Preliminaries
Create directories where model weights and logs will be saved.
```
mkdir weights
mkdir logs
```

## Data creation

```
python create_datasets.py --n_train 10000 --n_valid 1000 --n_test 1000 --n_gen 1000 --prefix length5_ninn_withholding --length 5 --vocab_size 10 --withheld 0-0,1-1,2-2,3-3,4-4
python list_all_5s.py 
```

## Train models

Generate training scripts

```
python scriptwriter_train_rnns_perfection.py
python scriptwriter_train_transformers_perfection.py
python scriptwriter_train_rnns_training_curve.py 
python scriptwriter_train_transformers_training_curve.py 
``` 

Now all of the training commands to be run will be in the following scripts, which call`main.py`. We do not recommend running these scripts themselves because each will take a very long time; instead, we recommend running all of the lines within a script in parallel.
```
scripts_rnns_perfection.sh
scripts_transformers_perfection.sh
scripts_runs_training_curve.sh
scripts_transformers_training_curve.sh
```

## Evaluate models
Evaluating on perfection: First create the perfection evaluation scripts.
```
python scriptwriter_eval_rnns_perfection.py
python scriptwriter_eval_transformers_perfection.py
```

And then run the scripts that have been created:
```
scripts_rnns_perfection_eval.sh
scripts_transformers_perfection_eval.sh
```

## Other files:

- `training.py`: Definition of the training procedure
- `rnn.py`, `transformer.py`, and `tp_transformer.py`: Definitions of the model architectures
- `data_loading.py`: For loading a dataset of integer sequences from a file
- `utils.py`: Contains miscellaneous helper functions

## Citation

To be added once available.


