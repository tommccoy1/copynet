

fo = open("scripts_transformers_training_curve.sh", "w")


for seed in range(10):
    for model in ["transformer", "tp_transformer"]:
        for training_size in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
            jobname = "training_curve_" + model + "_trainsize_" + str(training_size) + "_dataset_length5_ninn_withholding_bs_10_patience_5_hidden_256_230_heads_4_layers_2_filter_256_seed_" + str(seed)

            if model == "transformer":
                fo.write("time python main.py --model transformer  --dataset_prefix length5_ninn_withholding  --batch_size 10 --eval_every " + str(max(training_size // 10, 1)) + " --patience 5 --hidden 256 --vocab_size 10 --n_heads 4 --n_layers 2 --filter 256 --random_seed " + str(seed) + " --model_name " + jobname + " --training_size " + str(training_size) + " > logs/" + jobname + ".results" + "\n")
            else:
                fo.write("time python main.py --model tp_transformer  --dataset_prefix length5_ninn_withholding  --batch_size 10 --eval_every " + str(max(training_size // 10, 1)) + "  --patience 5 --hidden 230 --vocab_size 10 --n_heads 4 --n_layers 2 --filter 256 --random_seed " + str(seed) + " --model_name " + jobname + " --training_size " + str(training_size) + " > logs/" + jobname + ".results" + "\n")



