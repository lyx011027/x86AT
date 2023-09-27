from datetime import timedelta
import os
# 存放原始数据的文件夹
DATA_SOURCE_PATH = "/home/hw-admin/yixuan/data"
# 按sn号切分
SPLIT_DATA_PATH = os.path.join("split")
# 存放生成数据集的路径
DATA_SET_PATH = os.path.join("data_set")
# 存放测试模型的路径
TEST_MODEL_PATH = os.path.join("model_test")
# 存放测试模型R-P曲线图的文件夹
TEST_PIC_PATH = os.path.join("pic_test")
# 存放训练得到的数据集
MODEL_PATH =  os.path.join("model")
# 存放R-P曲线图的文件夹
PIC_PATH = os.path.join("pic")
# 提前预测时间，单位为minute
AHEAD_TIME_List = [timedelta(seconds=15),timedelta(minutes=1),timedelta(minutes=15),timedelta(minutes=30),timedelta(minutes=60), timedelta(hours=6)]
# 按batch生成数据集时，batch中dimm的数量，如果使用sample_batch.py生成数据集时发生OOM，则降低该值
BATCH_SIZE = 10000
MAXIMUM_RATIO = 100
STATIC_ITEM = [ "bit_width" ,"bit_width_x" ,"capacity"  ,'min_voltage',"part_number"   ,"procedure" ,"rank_count" ,"speed" ,"vendor"]


CETypeList = ['Corrected Errors']
UERTypeList = ['Uncorrected Error-SRAR', 'Uncorrected Error-Catastrophic/Fatal']
UEOTypeList = ['Uncorrected Error-SRAO', 'Uncorrected Error-UCNA']
UETypeList = UERTypeList + UEOTypeList

PatrolScrubbingUETypeList = ['Downgraded Uncorrected PatrolScrubbing Error']