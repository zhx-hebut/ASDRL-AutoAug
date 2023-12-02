import glob
import os
import csv
from PIL import Image
from torchvision.utils import save_image

train_path = '../train'
val_path = '../val'
test_path = '../test'

train_image = glob.glob(r'../train/img/*')
val_image = glob.glob(r'../val/image/*')
test_image = glob.glob(r'../test/image/*')

train = open('../train.csv','w',encoding='utf-8')
val = open('../val.csv','w',encoding='utf-8')
test = open('../test.csv','w',encoding='utf-8')

for i in train_image:
    name_ = os.path.split(i)[-1]
    name = name_.split('.',2)[0]
    # name_A = os.path.join(train_path,'img',name + '.jpg')
    name_A = os.path.join(train_path,'img',name_)
    name_B = os.path.join(train_path,'label',name + '.png')
    writer = csv.writer(train)
    writer.writerow([name_A,name_B])

for j in val_image:
    name_ = os.path.split(j)[-1]
    name = name_.split('.',2)[0]
    name_A = os.path.join(val_path,'image',name + '.png')
    name_B = os.path.join(val_path,'label',name + '.png')
    writer = csv.writer(val)
    writer.writerow([name_A,name_B])
    
for k in test_image:
    name_ = os.path.split(k)[-1]
    name = name_.split('.',2)[0]
    name_A = os.path.join(test_path,'image',name + '.png')
    name_B = os.path.join(test_path,'label',name + '.png')
    writer = csv.writer(test)
    writer.writerow([name_A,name_B])
print('end')