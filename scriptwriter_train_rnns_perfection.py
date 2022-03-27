

fo = open("scripts_rnns_perfection.sh", "w")


for seed in range(100):
    for model in ["gru", "lstm"]:
        jobname = "perfection_" + model + "_trainsize_10000_dataset_length5_ninn_withholding_bs_10_patience_5_hidden_296_257_layers_2_seed_" + str(seed)

        if model == "gru":
            fo.write("time python main.py --model gru  --dataset_prefix length5_ninn_withholding  --batch_size 10 --eval_every 1000 --patience 5 --hidden 296 --vocab_size 10 --n_layers 2 --random_seed " + str(seed) + " --model_name " + jobname + " > logs/" + jobname + ".results" + "\n")
        else:
            fo.write("time python main.py --model lstm  --dataset_prefix length5_ninn_withholding  --batch_size 10 --eval_every 1000  --patience 5 --hidden 257 --vocab_size 10 --n_layers 2 --random_seed " + str(seed) + " --model_name " + jobname + " > logs/" + jobname + ".results" + "\n")



