import os
import numpy as np
import random

def makeSplits():
    f = open("./mine_0322/makeSplits.txt", "w+")

    # '2011_09_26/2011_09_26_drive_0009_sync',这个数据集中缺少.bin真值
    pre_str = [ '2011_09_26/2011_09_26_drive_0018_sync',
                '2011_09_26/2011_09_26_drive_0051_sync',
                '2011_09_26/2011_09_26_drive_0056_sync',
                '2011_09_26/2011_09_26_drive_0059_sync',
                '2011_09_26/2011_09_26_drive_0093_sync',
                '2011_09_26/2011_09_26_drive_0096_sync',
                '2011_09_26/2011_09_26_drive_0104_sync',
                '2011_09_26/2011_09_26_drive_0117_sync',
                '2011_09_28/2011_09_28_drive_0039_sync',
                '2011_09_29/2011_09_29_drive_0071_sync']

    pic_nums = [447,270,438,294,373,433,475,312,660,352,1059]
    # print(len(pic_nums))

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


def genTestFiles():
    f = open("./mine_0322/test_files.txt", "w+")

    pre_str = [ '2011_09_26/2011_09_26_drive_0009_sync',
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

    pic_nums = [447,270,438,294,373,433,475,312,660,352,1059]
    # print(len(pic_nums))

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
    # genSplits()
    genTestFiles()