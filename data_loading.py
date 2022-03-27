



def load_dataset(filename):
    fi = open(filename, "r")

    dataset = []

    for line in fi:
        seq = [int(x) for x in line.strip().split()]

        dataset.append(seq)


    return dataset




