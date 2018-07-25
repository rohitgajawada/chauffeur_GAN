###########################################################
# Script for renaming .h5 files when merging two datasets #
###########################################################

import glob
from shutil import copyfile

d1 = glob.glob('/home/adas/Documents/CARLA_0.8.2_SantQuirze/PythonClient/LowQualityDataset2805/SeqTrain/')
d2 = glob.glob('/home/adas/Documents/CARLA_0.8.2_SantQuirze/PythonClient/LowQuality2xlap15fps/*.h5')
d3 = glob.glob('/home/adas/Documents/CARLA_0.8.2_SantQuirze/PythonClient/LowQuality2xrevlap15fps/*.h5')
d4 = glob.glob('/home/adas/Documents/CARLA_0.8.2_SantQuirze/PythonClient/LowQualityTurnsRev15fps/*.h5')

total = d2 + d3 + d4

num = 900
print (len(total))
for filename in total:
    filename_new = filename[-13:-7]
    num = num + 1
    if num / 1000 == 0:
        filename_new += '0' + str(num) + '.h5'
    else:
        filename_new += str(num) + '.h5'
    # print (filename_new)
    dest = '/home/adas/Documents/CARLA_0.8.2_SantQuirze/PythonClient/LowQualityDataset2805/'
    copyfile(filename, dest + filename_new)
