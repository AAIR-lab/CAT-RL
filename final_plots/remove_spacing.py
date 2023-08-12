filepath1 = "./results/acc_rewards/mountaincar_adrl_1.csv"
filepath2 = "./results/acc_rewards/mountaincar_adrl_2.csv"

f = open(filepath2, "w")

with open(filepath1, 'r') as f1:
    line = f1.read()
    lines = line.split("\n")
    for line in lines:
        print(line)
        if line!="":
            f.write(line)
            f.write("\n")


f.close()