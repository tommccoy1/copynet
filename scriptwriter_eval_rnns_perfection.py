

fo = open("scripts_rnns_perfection_eval.sh", "w")

for seed in range(100):
    for model in ["gru", "lstm"]:
        jobname = "perfection_" + model + "_trainsize_10000_dataset_length5_ninn_withholding_bs_10_patience_5_hidden_296_257_layers_2_seed_" + str(seed)

        if model == "gru":
            fo.write("time python eval_perfection.py --model gru  --dataset_prefix length5_ninn_withholding  --batch_size 10 --eval_every 1000 --patience 5 --hidden 296 --vocab_size 10 --n_layers 2 --random_seed " + str(seed) + " --model_name " + jobname + " --all_data_list all_5s.txt --print_errors > logs/" + jobname + ".perfection_results" + "\n")
        else:
            fo.write("time python eval_perfection.py --model lstm  --dataset_prefix length5_ninn_withholding  --batch_size 10 --eval_every 1000  --patience 5 --hidden 257 --vocab_size 10 --n_layers 2 --random_seed " + str(seed) + " --model_name " + jobname + " --all_data_list all_5s.txt --print_errors > logs/" + jobname + ".perfection_results" + "\n")



