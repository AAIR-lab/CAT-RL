import csv
import numpy as np

# with open("./acc_rewards/taxi_30x30_map1_adrl_1.csv","r") as f1:
#     with open("./acc_rewards/taxi_30x30_map1_adrl_11.csv","w") as f2:
#         csvreader = csv.reader(f1, delimiter=',')
#         writer = csv.writer(f2)
#         for item in csvreader:
#             writer.writerow(item)
#             break
#         csvreader = csv.reader(f1, delimiter=',')
#         for item in csvreader:
#             if item!=[]:
#                 writer.writerow([item[0], item[1]])

# for taking averages for officeworld PPO (due to high std dev)
episodes = []
data = {}
for i in range(1,11):
    with open("./results/officeworld/ppo/eval/eval_officeworld_ppo_"+str(i)+".csv","r") as f1:
        csvreader = csv.reader(f1, delimiter=',')    
        for j,item in enumerate(csvreader):
            if j == 0:
                continue
            if i == 1:
                episodes.append(item[0])
            vals = np.asarray([float(token) for token in item[1][2:-2].split(",")])
            if item[0] not in data:
                data[item[0]] = vals
            else:
                data[item[0]] += vals

s = "Num episodes,Cumulative rewards\n"
for i in episodes:
    s += str(i) + "," + '"' +str((data[i] / 10.0).tolist()) + '"' + "\n"

with open("./results/officeworld/ppo/eval/final.csv","w") as f:
    f.write(s)
    f.close()