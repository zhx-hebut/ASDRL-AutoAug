import os 
import csv
import time 
import numpy as np

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_csv(data_dir):
    list = []
    with open(data_dir,"r",encoding='UTF-8') as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            list.append(line) 
    return list  
 
def format_time():
    time_ = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    return time_

def tensor_to_image(img_tensor):

    img = img_tensor.permute(0,2,3,1).squeeze(0).squeeze(2).numpy() 
    return img  


# def read_csv(path):
#     # filename=input("请输入文件名： ")
#     # filename=input(path)
#     with open(path,'rt',encoding='UTF-8')as raw_data:
#         readers=csv.reader(raw_data,delimiter=',')
#         x=list(readers)
#         data=np.array(x)
#         # print(data)
#         # print(data.shape)
#     return data