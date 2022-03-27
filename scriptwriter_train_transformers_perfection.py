

fo = open("scripts_transformers_perfection.sh", "w")


for seed in range(100):
    for model in ["transformer", "tp_transformer"]:
        jobname = "perfection_" + model + "_trainsize_10000_dataset_length5_ninn_withholding_bs_10_patience_5_hidden_256_230_heads_4_layers_2_filter_256_seed_" + str(seed)

        if model == "transformer":
            fo.write("time python main.py --model transformer  --dataset_prefix length5_ninn_withholding  --batch_size 10 --eval_every 1000 --patience 5 --hidden 256 --vocab_size 10 --n_heads 4 --n_layers 2 --filter 256 --random_seed " + str(seed) + " --model_name " + jobname + " > logs/" + jobname + ".results" + "\n")
        else:
            fo.write("time python main.py --model tp_transformer  --dataset_prefix length5_ninn_withholding  --batch_size 10 --eval_every 1000 --patience 5 --hidden 230 --vocab_size 10 --n_heads 4 --n_layers 2 --filter 256 --random_seed " + str(seed) + " --model_name " + jobname + " > logs/" + jobname + ".results" + "\n")



