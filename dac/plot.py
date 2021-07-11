import matplotlib.pyplot as plt
import numpy as np
import glob

data=[]

for name in glob.glob("logs/reward_ecc_[0-9]*.txt"):
    with open(name,'r') as file:
        curr_data=[]
        for i,line in enumerate(file):
            line=line.strip()
            if(line=="" or i==0):
                continue
            curr_data.append(float(line))
        data.append(curr_data)

data=np.array(data)
print(data.shape)

average_rewards=np.sum(data,axis=0)/data.shape[0]
# average_rewards=average_rewards[35:]
plt.plot(average_rewards)
plt.ylabel("average rewards of all households")
plt.xlabel("episodes")
plt.show()
names=[]
for i,row in enumerate(data):
    plt.plot(row)
    names.append(f"ecc-{i}")
plt.legend(names)
plt.xlabel("episodes")
plt.ylabel("reward")
plt.show()