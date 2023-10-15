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



CETypeList = ['Corrected Error']
UERTypeList = ['Uncorrected Error-SRAR', 'Uncorrected Error-Catastrophic/Fatal']
UEOTypeList = ['Uncorrected Error-SRAO', 'Uncorrected Error-UCNA']
UETypeList = UERTypeList + UEOTypeList

PatrolScrubbingUETypeList = ['Downgraded Uncorrected PatrolScrubbing Error']

STATIC_ITEM = ['bit_width' ,'bit_width_x' ,'capacity'   ,'min_voltage' ,'part_number'  ,'rank_count' ,'speed'    ,'technology','type' ,'vendor']
OBSERVATION_TIME_LIST = [timedelta(minutes=6), timedelta(hours=1),timedelta(hours=6), timedelta(hours=24), timedelta(hours=72), timedelta(hours=120)]
# OBSERVATION_TIME_LIST = [timedelta(minutes=1), timedelta(minutes=5), timedelta(hours=1), timedelta(hours=3), timedelta(hours=12), timedelta(hours=24)]

# 提前预测时间
LEAD_TIME_LIST = [timedelta(seconds=1),timedelta(seconds=30),timedelta(minutes=1),timedelta(minutes=5),timedelta(minutes=60)]


def getMinutes(time):
    return int(time.days * 24 * 60 + time.seconds / 60)
CEIntervalNumList = [3, 5, 7] 
FltCnt = {'Cell':2,'Row':2,'Column':2,'Bank':3,'Device':2}


def getDynamicSample():
    sample = {}
    sample = getFrequencySample(sample)
    sample = getBitLevelSample(sample)
    sample = getSubBankSample(sample)
    sample = getCECountSample(sample)
    return sample


def getFrequencySample(sample):
    for time in OBSERVATION_TIME_LIST:
        # sample['Rpt_CE_Cnt_{}'.format(getMinutes(time))] = -1
        sample['Err_CE_Cnt_{}'.format(getMinutes(time))] = -1
        for num in CEIntervalNumList:
            sample['Min_T_CE_{}_{}'.format(getMinutes(time), num)] = -1
    # for level in FltCnt.keys():
    #     sample[level] = False
    sample['lifeSpan'] = -1
    sample['errorSpan'] = -1
    sample['errorAvg'] = -1
    return sample

prefixList = ['max', 'sum', 'min', 'avg']
patternList = ['adjERRDq', 'maxDqDistance','minDqDistance', 'maxBurstDistance','minBurstDistance', 'errorBurst', 'errorDq', 'errorPerBurst']
def getBitLevelSample(sample):
    for prefix in prefixList:
        for pattern in patternList:
            sample['{}_{}'.format(prefix,pattern)] = 0
    return sample
bitLevelItem = list(getBitLevelSample({}).keys())
def getSubBankSample(dynamicSample):

    dynamicSample['subBank_count'] = 0
    dynamicSample['subBank_avg'] = 0
    dynamicSample['subBank_max'] = 0
    return dynamicSample

def getCECountSample(dynamicSample):

    for level in FltCnt.keys():
        dynamicSample['{}_count'.format(level)] = 0
        dynamicSample['{}_avg'.format(level)] = 0    
        dynamicSample['{}_max'.format(level)] = 0   
    dynamicSample['noadd'] = 0
    dynamicSample['CE_number'] = 0
    dynamicSample['PatrolScrubbingUEO'] = 0
    return dynamicSample
      
dynamicItem = list(getDynamicSample().keys())
LEAD = timedelta(minutes=0)
sampleDistance = 5
dataSetFile = "{}.csv".format(sampleDistance)


subBankTime = timedelta(minutes=5)
OBSERVATION = timedelta(hours=120)
Predict = timedelta(days=30)
Interval = timedelta(minutes=5)

