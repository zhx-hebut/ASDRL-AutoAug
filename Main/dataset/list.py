import os
from PIL import Image

images_path = '../imgs'
masks_path = '../masks'

train_path = '../img_train.list'
val_path = '../img_val.list'
test_path = '../img_test.list'

save_train_path = '../train'
save_val_path = '../val'
save_test_path = '../test'

with open(test_path) as f:
    name_list = f.readlines()
    for i in name_list:
        name = i.strip('\n')
        image_path = os.path.join(images_path,name)
        label_path = os.path.join(masks_path,name)
        image_open = Image.open(image_path)
        label_open = Image.open(label_path)
        image_open.save(os.path.join(save_test_path,'image',name))
        label_open.save(os.path.join(save_test_path,'label',name))
    print('1')