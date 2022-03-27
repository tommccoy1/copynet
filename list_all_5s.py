
# Lists all sequences of length 5

fo = open("data/all_5s.txt", "w")


for i in range(100000):
    string_num = str(i)

    while len(string_num) != 5:
        string_num = "0" + string_num

    list_num = list(string_num)

    fo.write(" ".join(list_num) + "\n")





