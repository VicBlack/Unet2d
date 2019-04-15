import os
import random


def travel_files(file_path):
    file_items = []
    for root, dirs, files in os.walk(file_path, topdown=True):
        for name in files:
            if name.find('SA') > 0 and name.find('dcm') > 0:
                file_items.append((os.path.join(root, name), os.path.join(file_path, 'masks', name[:-3]+'png')))
    random.shuffle(file_items)
    return file_items

def travel_testfiles(file_path):
    file_items = []
    for root, dirs, files in os.walk(file_path, topdown=True):
        for name in files:
            if name.find('SA') > 0 and name.find('dcm') > 0:
                file_items.append(os.path.join(root, name))
    random.shuffle(file_items)
    return file_items

def data_set_split(file_items):
    partition = {}
    holdout_percentage = 0.1
    partition['holdout'] = file_items[0: int(len(file_items)*holdout_percentage)]
    train_list = file_items[int(len(file_items)*holdout_percentage): len(file_items)]

    train_percentage = 0.8
    partition['train'] = train_list[0:int(len(train_list)*train_percentage)]
    partition['test'] = train_list[int(len(train_list)*train_percentage): len(train_list)]

    #print(partition)
    return partition



if __name__=='__main__':
    file_path = 'E:/DATA/DCMS/'
    file_items = travel_files(file_path)
    data_set_split(file_items)




