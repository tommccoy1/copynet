import sys
import os

all_architectures_test = {}
all_architectures_gen = {}

size_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000] 

for architecture in ["lstm", "gru", "transformer", "tp_transformer"]:

    test_acc_list = []
    gen_acc_list = []

    for trainsize in size_list:

        print(trainsize)

        test_accs = []
        gen_accs = []

        for seed in range(10):
            if "transformer" in architecture:
                fi = open("logs/training_curve_" + architecture + "_trainsize_" + str(trainsize) + "_dataset_length5_ninn_withholding_bs_10_patience_5_hidden_256_230_heads_4_layers_2_filter_256_seed_" + str(seed) + ".results", "r")
            else:
                fi = open("logs/training_curve_" + architecture + "_trainsize_" + str(trainsize) + "_dataset_length5_ninn_withholding_bs_10_patience_5_hidden_296_257_layers_2_seed_" + str(seed) + ".results", "r")


            for line in fi:
                if line.startswith("Test set:"):
                    parts = line.strip().split()
                    acc = parts[2]
                    test_accs.append(float(acc))

                if line.startswith("Gen set:"):
                    parts = line.strip().split()
                    acc = parts[2]
                    gen_accs.append(float(acc))


        total_test_acc = 0
        count_test_acc = 0

        for elt in test_accs:
            total_test_acc += elt
            count_test_acc += 1

        test_acc_list.append(total_test_acc*1.0/count_test_acc)

        total_gen_acc = 0
        count_gen_acc = 0

        for elt in gen_accs:
            total_gen_acc += elt
            count_gen_acc += 1

        gen_acc_list.append(total_gen_acc*1.0/count_gen_acc)


    all_architectures_test[architecture] = test_acc_list
    all_architectures_gen[architecture] = gen_acc_list

for architecture in all_architectures_test:
    print(architecture)
    print(all_architectures_test[architecture])
    #print(list(zip(size_list, all_architectures_test[architecture])))
    print("")


for architecture in all_architectures_gen:
    print(architecture)
    print(all_architectures_gen[architecture])
    #print(list(zip(size_list,all_architectures_gen[architecture])))
    print("")




