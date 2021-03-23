import os
import numpy as np
import random
import math

# save_path
split_folder_path = "./split0323/"
split_file_path = split_folder_path + "split.txt"
train_file_path = split_folder_path + "train_files.txt"
val_file_path = split_folder_path + "val_files.txt"
test_file_path = split_folder_path + "test_files.txt"

# data_lists
pre_str = ['2011_09_26/2011_09_26_drive_0009_sync',
           '2011_09_26/2011_09_26_drive_0018_sync',
           '2011_09_26/2011_09_26_drive_0051_sync',
           '2011_09_26/2011_09_26_drive_0056_sync',
           '2011_09_26/2011_09_26_drive_0059_sync',
           '2011_09_26/2011_09_26_drive_0093_sync',
           '2011_09_26/2011_09_26_drive_0096_sync',
           '2011_09_26/2011_09_26_drive_0104_sync',
           '2011_09_26/2011_09_26_drive_0117_sync',
           '2011_09_28/2011_09_28_drive_0039_sync',
           '2011_09_29/2011_09_29_drive_0071_sync']

pic_nums = [447, 270, 438, 294, 373, 433, 475, 312, 660, 352, 1059]

def genSplits():
    f = open(split_file_path, "w+")

    nums = []
    for i in range(0,len(pic_nums)):
        rang = np.arange(0, pic_nums[i])
        np.random.shuffle(rang)
        nums.append(rang)

    idx_arr = np.arange(0,len(pic_nums))
    np.random.shuffle(idx_arr)

    for i,idx in enumerate(idx_arr):
        print(idx,'\n')
        for j,v in enumerate(nums[idx]):
            file_str = pre_str[idx]+' '+ str(v)
            if (i+j)%2==0:
                file_str +=' r\n'
            else:
                file_str +=' l\n'
            f.write(file_str)

    f.close()


def genTrainAndTestFiles(shuffle=False):
    f = open(split_file_path, "r")
    train_file =  open(train_file_path, "w+")
    val_file =  open(val_file_path, "w+")

    # 读取文本名字并打乱
    data = f.readlines()
    if shuffle:
        np.random.shuffle(data)
    file_num = len(data)
    ratio = 0.75

    # 前75%存入train_file
    for i in range(0,file_num):
        if i<=ratio*file_num:
            train_file.write(data[i])
        else:
            val_file.write(data[i])

    f.close()
    train_file.close()
    val_file.close()

def genTestFiles():
    f = open(test_file_path, "w+")

    nums = []
    for i in range(0,len(pic_nums)):
        rang = np.arange(0, pic_nums[i])
        # np.random.shuffle(rang)
        nums.append(rang)

    idx_arr = np.arange(0,len(pic_nums))
    # np.random.shuffle(idx_arr)

    for i,idx in enumerate(idx_arr):
        print(idx,'\n')
        for j,v in enumerate(nums[idx]):
            file_str = (pre_str[idx]+' {:010d}').format(v)
            file_str +=' l\n'
            f.write(file_str)

    f.close()

if __name__=="__main__":
    genSplits()
    print("Gen splits text.\n")

    genTrainAndTestFiles()
    print("Gen TrainAndTest text.\n")

    genTestFiles()
    print("Gen splits text.\n")