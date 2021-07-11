import csv
import re
import numpy as np
from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer
import threading
import argparse

filename = './15minute_data_newyork/15minute_data_newyork.csv'
my_regex = re.compile(r'([a-zA-Z]*)([0-9]*)')
NSA = 0
NIA = 1
IA = 2
appliance_type = {
    'air': IA,
    'airwindowunit': IA,
    'aquarium': NSA,
    'bathroom': NSA,
    'battery': IA,
    'bedroom': NSA,
    'car': IA,
    'circpump': NSA,
    'clotheswasher': NIA,
    'diningroom': NSA,
    'dishwasher': NIA,
    'disposal': NSA,
    'drye': NIA,
    'dryg': NIA,
    'freezer': NSA,
    'furnace': IA,
    'garage': NSA,
    'heater': NSA,
    'housefan': NSA,
    'icemaker': NSA,
    'jacuzzi': NSA,
    'kitchen': NSA,
    'kitchenapp': NSA,
    'lights': NSA,
    'livingroom': NSA,
    'microwave': NSA,
    'office': NSA,
    'outsidelights': NSA,
    'oven': NSA,
    'pool': NSA,
    'poollight': NSA,
    'poolpump': NSA,
    'range': NSA,
    'refrigerator': NSA,
    'security': NSA,
    'sewerpump': NSA,
    'shed': NSA,
    'sprinkler': NSA,
    'sumppump': NSA,
    'utilityroom': NSA,
    'waterheater': NSA,
    'wellpump': NSA,
    'winecooler': NSA
}
nsa_cols=[]
nia_cols=[]
ia_cols=[]
rows=[]
data={}
time_dict={}
house_ids=[]
counter_id={}
total_size=0
read_completed=False
def read_data(num_households):
    with open(filename, 'r') as csvfile:
        global nsa_cols
        global nia_cols
        global ia_cols
        global rows
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for i,field in enumerate(fields):
            ans = my_regex.search(field).groups()
            if ans[0] in appliance_type.keys():
                app_type=appliance_type[ans[0]]
                if app_type==NSA:
                    nsa_cols.append(i)
                elif app_type==NIA:
                    nia_cols.append(i)
                elif app_type==IA:
                    ia_cols.append(i)
        nsa_cols=np.array(nsa_cols)
        nia_cols=np.array(nia_cols)
        ia_cols=np.array(ia_cols)
        # count=10000
        for row in csvreader:
            rows.append(row)
            # count-=1
            # if count==0:
            #     break
        rows=np.array(rows)

    ia_data=rows[:,ia_cols]
    nia_data=rows[:,nia_cols]
    nsa_data=rows[:,nsa_cols]

    
    for row in rows[:500]:
        global house_ids
        house_id=int(row[0])
        data[house_id]=[]
        time_dict[house_id]=1
        counter_id[house_id]=0
        house_ids.append(house_id)

    house_ids=np.unique(house_ids)

    with open("participants.txt","w") as file:
        ecc = []
        for i,house_id in enumerate(house_ids[:num_households]):
            ecc.append(f"127.0.0.1 {3000+i} {i}\n")
        file.writelines(ecc)
        print(f"Number of ecc's:{len(ecc)}")

    for counter,row in enumerate(rows):
        house_id=int(row[0])
        temp=row[1].split(" ")
        time_split=temp[1].split(":")
        actual_time_key=":".join([temp[0],time_split[0],time_split[1]])
        counter_id[house_id]+=1
        timestamp=time_dict[house_id]
        time_dict[house_id]+=1
        if counter_id[house_id]==144:
            time_dict[house_id]=1
            counter_id[house_id]=0
        t1=ia_data[counter]
        t1=t1[t1!=""].astype('float64')
        t2=nia_data[counter]
        t2=t2[t2!=""].astype('float64')
        t3=nsa_data[counter]
        t3=t3[t3!=""].astype('float64')
        data[house_id].append([timestamp,list(t3),list(t2),list(t1),actual_time_key])


    for i in range(len(house_ids[:num_households])):
        global total_size
        h_id=house_ids[i]
        data[h_id].sort(key=lambda x:x[-1])
        print(data[h_id][:10])
        for row in data[h_id]:
            del(row[-1])
        temp=data[h_id][0]
        curr_size=1
        for l in temp[1:]:
            curr_size+=2*len(l)
        curr_size*=2
        total_size+=curr_size
        # print(total_size)


    print("Completed the Read")
    global read_completed
    read_completed=True
def get_data(house_id):
    # print(house_ids[house_id])
    return data[house_ids[house_id]]

def get_total_size():
    # return [total_size],len(data.keys())
    return [total_size],num_households
def get_server_status():
    return read_completed

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_households", help="Enter Number of Households for experiment")
    args = parser.parse_args()
    num_households = int(args.num_households)
    t = threading.Thread(target=read_data, args=(num_households,))
    t.start()
    server = SimpleJSONRPCServer(('0.0.0.0', 8000))
    server.register_function(get_data)
    server.register_function(get_total_size)
    server.register_function(get_server_status,'server_status')
    server.serve_forever()