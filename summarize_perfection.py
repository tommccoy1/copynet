import sys
import os

all_architectures_0dev = {}
all_architectures_1dev = {}
all_architectures_2dev = {}
all_architectures_3dev = {}
all_architectures_4dev = {}
all_architectures_5dev = {}
all_architectures_6dev = {}

all_architectures_correct = {}
all_architectures_insertion = {}
all_architectures_deletion = {}
all_architectures_substitution = {}
all_architectures_swap = {}
all_architectures_other = {}

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

    all_architectures_correct[architecture] = []
    all_architectures_insertion[architecture] = []
    all_architectures_deletion[architecture] = []
    all_architectures_substitution[architecture] = []
    all_architectures_swap[architecture] = []
    all_architectures_other[architecture] = []

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

            if line.startswith("correct") and len(line.strip().split()) == 2:
                all_architectures_correct[architecture].append(int(line.strip().split()[1]))

            if line.startswith("insertion") and len(line.strip().split()) == 2:
                all_architectures_insertion[architecture].append(int(line.strip().split()[1]))

            if line.startswith("deletion") and len(line.strip().split()) == 2:
                all_architectures_deletion[architecture].append(int(line.strip().split()[1]))

            if line.startswith("substitution") and len(line.strip().split()) == 2:
                all_architectures_substitution[architecture].append(int(line.strip().split()[1]))

            if line.startswith("swap") and len(line.strip().split()) == 2:
                all_architectures_swap[architecture].append(int(line.strip().split()[1]))

            if line.startswith("other") and len(line.strip().split()) == 2:
                all_architectures_other[architecture].append(int(line.strip().split()[1]))




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
    count_errors = sum(all_architectures_insertion[architecture]) + sum(all_architectures_deletion[architecture]) + sum(all_architectures_substitution[architecture]) + sum(all_architectures_swap[architecture]) + sum(all_architectures_other[architecture])
    print("Insertion", sum(all_architectures_insertion[architecture]), sum(all_architectures_insertion[architecture])*1.0/count_errors)
    print("Deletion", sum(all_architectures_deletion[architecture]), sum(all_architectures_deletion[architecture])*1.0/count_errors)
    print("Substitution", sum(all_architectures_substitution[architecture]), sum(all_architectures_substitution[architecture])*1.0/count_errors)
    print("Swap", sum(all_architectures_swap[architecture]), sum(all_architectures_swap[architecture])*1.0/count_errors)
    print("Other", sum(all_architectures_other[architecture]), sum(all_architectures_other[architecture])*1.0/count_errors)
    print("Total errors", count_errors)
    print("")


for architecture in architectures:
    if count_perfection[architecture][1] == 0:
        continue
    print(architecture)
    print("Perfection:", count_perfection[architecture][0]*1.0/count_perfection[architecture][1], count_perfection[architecture][0], count_perfection[architecture][1])
    print("")




