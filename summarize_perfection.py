import sys
import os

all_architectures_0dev = {}
all_architectures_1dev = {}
all_architectures_2dev = {}
all_architectures_3dev = {}
all_architectures_4dev = {}
all_architectures_5dev = {}
all_architectures_6dev = {}

count_perfection = {}

architectures = ["lstm", "gru", "transformer", "tp_transformer"]

for architecture in architectures:
    all_architectures_0dev[architecture] = []
    all_architectures_1dev[architecture] = []
    all_architectures_2dev[architecture] = []
    all_architectures_3dev[architecture] = []
    all_architectures_4dev[architecture] = []
    all_architectures_5dev[architecture] = []
    all_architectures_6dev[architecture] = []

    count_perfection[architecture] = [0,0]


    for seed in range(100):
        if "transformer" in architecture:
            fi = open("logs/perfection_" + architecture + "_trainsize_10000_dataset_length5_ninn_withholding_bs_10_patience_5_hidden_256_230_heads_4_layers_2_filter_256_seed_" + str(seed) + ".perfection_results", "r")
        else:
            fi = open("logs/perfection_" + architecture + "_trainsize_10000_dataset_length5_ninn_withholding_bs_10_patience_5_hidden_296_257_layers_2_seed_" + str(seed) + ".perfection_results", "r")

        total_correct = 0
        total_total = 0

        for line in fi:
            if "deviations" in line:
                parts = line.strip().split()
                acc = float(parts[2])
                count_correct = int(parts[3])
                count_total = int(parts[4])

                total_correct += count_correct
                total_total += count_total

                if line[0] == "0":
                    all_architectures_0dev[architecture].append(acc)
                if line[0] == "1":
                    all_architectures_1dev[architecture].append(acc)
                if line[0] == "2":
                    all_architectures_2dev[architecture].append(acc)
                if line[0] == "3":
                    all_architectures_3dev[architecture].append(acc)
                if line[0] == "4":
                    all_architectures_4dev[architecture].append(acc)
                if line[0] == "5":
                    all_architectures_5dev[architecture].append(acc)
                if line[0] == "6":
                    all_architectures_6dev[architecture].append(acc)

        if total_total == 100000:
            if total_correct == total_total:
                count_perfection[architecture][0] += 1
            count_perfection[architecture][1] += 1


def mean(lst):
    total = 0
    count = 0

    for elt in lst:
        total += elt
        count += 1

    if count == 0:
        return 0
    return total*1.0/count

for architecture in architectures:
    print(architecture)
    for i in range(6):
        print(str(i) + " deviations:", mean([all_architectures_0dev, all_architectures_1dev, all_architectures_2dev, all_architectures_3dev, all_architectures_4dev, all_architectures_5dev, all_architectures_6dev][i][architecture]))
    print("")


for architecture in architectures:
    if count_perfection[architecture][1] == 0:
        continue
    print(architecture)
    print("Perfection:", count_perfection[architecture][0]*1.0/count_perfection[architecture][1], count_perfection[architecture][0], count_perfection[architecture][1])
    print("")




