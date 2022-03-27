
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", help="size of the training set", type=int, default=None)
parser.add_argument("--n_valid", help="size of the validation set", type=int, default=None)
parser.add_argument("--n_test", help="size of the test set", type=int, default=None)
parser.add_argument("--n_gen", help="size of the generalization set", type=int, default=None)
parser.add_argument("--prefix", help="prefix of file to save the sequences to", type=str, default=None)
parser.add_argument("--length", help="sequence length", type=int, default=None)
parser.add_argument("--vocab_size", help="vocabulary size", type=int, default=10)
parser.add_argument("--withheld", help="comma-separated list of withheld digit/position pairs; each pair should be digit-hyphen-position. E.g., 0-1,2-3 means to withhold 0 in position 1 and 2 in position 3", type=str, default="")

args = parser.parse_args()


import random
valid_set = []
test_set = []
gen_set = []
training_set = []

used = {}

vocab = []
for i in range(args.vocab_size):
    vocab.append(i)


withhold_dict = {}
if len(args.withheld) != 0:
    withhold_pairs = args.withheld.split(",")
else:
    withhold_pairs = []

for pair in withhold_pairs:
    parts = pair.split("-")
    digit = int(parts[0])
    index = int(parts[1])

    if index not in withhold_dict:
        withhold_dict[index] = []
    if digit not in withhold_dict[index]:
        withhold_dict[index].append(digit)

def keep(seq):
    for index, elt in enumerate(seq):
        if index in withhold_dict and elt in withhold_dict[index]:
            return False
        
    return True

print("start")
for i in range(args.n_valid):
    satisfied = False
    while not satisfied:
        seq = []
        for i in range(args.length):
            seq.append(random.choice(vocab))
        if tuple(seq) not in used and keep(seq):
            satisfied = True
        
    valid_set.append([seq])
    used[tuple(seq)] = True
    
print("valid done")
for i in range(args.n_test):
    satisfied = False
    while not satisfied:
        seq = []
        for i in range(args.length):
            seq.append(random.choice(vocab))
        if tuple(seq) not in used and keep(seq):
            satisfied = True
        
    test_set.append([seq])
    used[tuple(seq)] = True   

print("test done")

if args.withheld != "":
    for i in range(args.n_gen):
        satisfied = False
        while not satisfied:
            seq = []
            for i in range(args.length):
                seq.append(random.choice(vocab))
            if tuple(seq) not in used and not keep(seq):
                satisfied = True
        
        gen_set.append([seq])
        used[tuple(seq)] = True   

for i in range(args.n_train):
    satisfied = False
    while not satisfied:
        seq = []
        for i in range(args.length):
            seq.append(random.choice(vocab))
        if tuple(seq) not in used and keep(seq):
            satisfied = True
        
    training_set.append([seq])
    used[tuple(seq)] = True 




fo_train = open("data/" + args.prefix + ".train", "w")
fo_valid = open("data/" + args.prefix + ".valid", "w")
fo_test = open("data/" + args.prefix + ".test", "w")
fo_gen = open("data/" + args.prefix + ".gen", "w")

for elt in training_set:
    fo_train.write(" ".join([str(x) for x in elt[0]]) + "\n")

for elt in valid_set:
    fo_valid.write(" ".join([str(x) for x in elt[0]]) + "\n")

for elt in test_set:
    fo_test.write(" ".join([str(x) for x in elt[0]]) + "\n")

for elt in gen_set:
    fo_gen.write(" ".join([str(x) for x in elt[0]]) + "\n")






