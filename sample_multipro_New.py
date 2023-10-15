import pandas as pd
from datetime import datetime, timedelta
import csv
from config import *
import os
import math
import copy
from multiprocessing import Process, Queue
# import moxing as mox
MAXROW = 20*8
MAXCOLUMN = 30


bitFlag = False
capacityFlag = False
bit_width_x = 4
capacity = 32*1024

staticItem = [
    'dimm_sn'
    ] + STATIC_ITEM



sampleItem = staticItem + dynamicItem  + ['time','label']



if not os.path.exists(DATA_SET_PATH):
    os.makedirs(DATA_SET_PATH)


def get_writer(dataset):
    f1 = open(dataset, mode="w")
    writer = csv.DictWriter(f1, sampleItem)
    itemMap = {}
    for item in sampleItem:
        itemMap [item] = item
    writer.writerow(itemMap)
    return writer

# 通过静态信息生成basesample

def getBaseSample(dimm, staticFile):
    staticDf = pd.read_csv(staticFile)
    sample = getDynamicSample()
    row = staticDf.loc[0]
    flag = False
    for item in STATIC_ITEM:
        
        sample[item] = row[item]
    if 'LifeStartDate' not in row:
        lifeStart = 0
        flag = False
    else:
        lifeStart = datetime.strptime(row['LifeStartDate'],'%Y-%m-%d %H:%M:%S')
        flag = True
    
    sample['dimm_sn'] = dimm
    
    return sample, lifeStart, flag


def addSubBankSample(sample, centerList, CEList):
    if len(centerList) == 0:
        sample['subBank_count'] = -1
        sample['subBank_avg'] = -1
        sample['subBank_max'] = -1
        return sample

    centerErrorList = len(centerList) * [0]
    for i in range(len(centerList)):
        center = centerList[i]
        centerTime, centerPosition, centerBankId = center
        
        for  error in CEList:
            bankId = "{}_{}_{}".format(error["rank"], error['bankgroup'], error['bank'])
            position = (error["row"],error["column"])
            errorTime = error['record_date']
            
            if (centerTime > errorTime
                and abs(centerPosition[0] - position[0]) < MAXROW 
                and abs(centerPosition[1] - position[1]) < MAXCOLUMN
                and centerBankId == bankId):

                centerErrorList[i] += error["err_count"]  
    sample['subBank_count'] = len(centerList)
    sample['subBank_avg'] = sum(centerErrorList) / len(centerErrorList)
    sample['subBank_max'] = max(centerErrorList)
    return sample

def addCenter(centerList, error):
    bankId = "{}_{}_{}".format(error["rank"], error['bankgroup'], error['bank'])
    position = (error["row"],error["column"])
    errorTime = error['record_date']
    # 新的 record 与 上一个 window 开始时间的间隔超过 interval，则创建一个新的 window
    
    flagNewCenter = True
    for center in centerList:
        if (center[0] > errorTime
            and abs(center[1][0] - position[0]) < MAXROW 
            and abs(center[1][1] - position[1]) < MAXCOLUMN
            and center[2] == bankId):
            flagNewCenter = False
            break
    if flagNewCenter:
        centerList.append([errorTime +subBankTime,position,bankId])
        
    return centerList

def addCECount(sample, bankGroupMap,CE_number):
    levelMap = {}
    for level in FltCnt.keys():
        levelMap[level] = []
    for bankgroup in bankGroupMap.values():
            for bank in bankgroup.values():
                rowMap = {}
                columnMap = {}
                for position in bank.keys():
                    count = bank[position]
                    levelMap['Cell'].append(count)
                    rowId, columnId = position
                        
                        
                    if rowId not in rowMap:
                        rowMap[rowId] = 0
                    if columnId not in columnMap:
                        columnMap[columnId] = 0
                        
                    rowMap[rowId] += 1
                    columnMap[columnId] += 1
                    
                for count in rowMap.values():
                    levelMap['Row'].append(count)
                for count in columnMap.values():
                    levelMap['Column'].append(count)
                
    for level in FltCnt.keys():
        if len(levelMap[level]) > 0:
            sample['{}_count'.format(level)] = len(levelMap[level])
            sample['{}_avg'.format(level)] = sum(levelMap[level])/len(levelMap[level])
            sample['{}_max'.format(level)] = max(levelMap[level])
    sample['CE_number'] = CE_number
    return sample




def addFrequency(sample, timeList):
    timeListLength = len(timeList)
    endTime = timeList[timeListLength - 1]
    
        
    for time in OBSERVATION_TIME_LIST:
        startTime = endTime - time
        for start in range(timeListLength):
            if timeList[start] >= startTime:
                break
        errorCount = timeListLength - start + 1
        sample['Err_CE_Cnt_{}'.format(getMinutes(time))] = errorCount
        for num in CEIntervalNumList:
            if num > errorCount:
                continue
            MinT =  int((timeList[timeListLength-1] - timeList[start]).total_seconds())
            for indx in range(start, timeListLength - errorCount):
                tmp = int((timeList[indx + errorCount -1] - timeList[indx]).total_seconds())
                MinT = min(tmp, MinT)
            sample['Min_T_CE_{}_{}'.format(getMinutes(time), num)] = MinT
    return sample
      
def processDimm(id, q, dimmList, leadTime):
    for dimm in dimmList:
        # print(dimm)
        staticFile = os.path.join(SPLIT_DATA_PATH, dimm, dimm+"_static.csv")
        # 生成静态信息
        baseSample, lifeStart ,Sflag= getBaseSample(dimm, staticFile)
        
        errorFile = os.path.join(SPLIT_DATA_PATH, dimm, dimm+"_error.csv")
       
        df = pd.read_csv(errorFile, low_memory=False)
        df['record_date'] = pd.to_datetime(df['record_date'], format="%Y-%m-%d %H:%M:%S")
        
        UEFlag = False
        firstUER = datetime.now().replace(year=2099)
        UEDf = df[(df['err_type'].isin(UETypeList))].reset_index(drop=True)
        if UEDf.shape[0] != 0:
            UEFlag = True
            firstUER = UEDf.loc[0, 'record_date']
        CEDf = df[(df['record_date'] <  firstUER) & (df['err_type'].isin(CETypeList))].reset_index(drop=True)
        
        baseSample['label'] = UEFlag
        
        CECount = CEDf.shape[0]
        if CECount < sampleDistance:
            continue
        
        errorStart = CEDf.loc[0, 'record_date']
        # 故障记录
        sampleList = []

        timeList = []

        
        bankGroupMap = {}
        CE_number = 0
        centerList = []
        CEList = []
        count = 0
        accumulateCE = 1
        for  index, error in CEDf.iterrows():
            errorTime = error['record_date']
            
            
            flag = False
            for i in range(len(timeList)):
                if errorTime - timeList[i] < OBSERVATION:
                    break
                else:
                    flag = True
                    
            if flag:
                timeList = timeList[i+1:]

            timeList.append(errorTime)
  
            CE_number += 1
            # if error['with_phy_addr']:
            rowId, columnId, bankId, bankgroupId =  error['row'], error['column'], error['bank'], error['bankgroup']
            if error['with_phy_addr']: 
                if bankgroupId not in bankGroupMap:
                    bankGroupMap[bankgroupId] = {}
                
                if bankId not in bankGroupMap[bankgroupId]:
                    bankGroupMap[bankgroupId][bankId] = {}

                position = (rowId,columnId) 
                if position not in bankGroupMap[bankgroupId][bankId]:
                    bankGroupMap[bankgroupId][bankId][position] = 0
                bankGroupMap[bankgroupId][bankId][position] += 1
                
                centerList = addCenter(centerList, error)
            else:
                baseSample['noadd'] += 1
                
            if accumulateCE < sampleDistance:
                accumulateCE += 1
                continue
            accumulateCE = 1

            count += 1
            sample = copy.copy(baseSample)
            sample['time'] = errorTime.timestamp()
            if Sflag:
                sample['lifeSpan'] = int((errorTime - lifeStart).total_seconds())
            sample['errorSpan'] = int((errorTime - errorStart).total_seconds())
            sample['errorAvg'] = sample['errorSpan'] / CE_number

            
            sample = addSubBankSample(sample, centerList, CEList)
            
            sample = addCECount(sample, bankGroupMap,CE_number)
            
            sample = addFrequency(sample, timeList)
            
            sampleList.append(sample)
            
        q.put([True, sampleList])   
            
            

def mergeFunction(q):
    writer = get_writer(os.path.join(DATA_SET_PATH,dataSetFile))
    while True:
        [flag , sampleList] = q.get()
        if not flag:
            break
        [writer.writerow(sample) for sample in sampleList]
        
def genDataSet(leadTime):
    
    # DATA_SET_PATH = 'train'
    if not os.path.exists(DATA_SET_PATH):
        os.makedirs(DATA_SET_PATH)
    
    dimmList = os.listdir(SPLIT_DATA_PATH)
    # dimmList = dimmList[10:103]
    q = Queue()
    processList = []
    cpuCount = os.cpu_count() * 2
    # cpuCount = 1
    subListSize = math.ceil(len(dimmList) / cpuCount)
    for i in range(cpuCount):
        subDimm = dimmList[i*subListSize:(i + 1)*subListSize]
        processList.append(Process(target=processDimm, args=(i,q, subDimm, leadTime)))
        
    pMerge = Process(target=mergeFunction, args=[q])
    pMerge.start()
    for p in processList:
        p.start()

    for p in processList:
        p.join()
    q.put([False,[]])
    pMerge.join()
    
    trainFile = os.path.join(DATA_SET_PATH, dataSetFile)
    trainDf = pd.read_csv(trainFile, low_memory=False)


    for item in STATIC_ITEM:
        trainDf["{}".format(item)] = pd.Categorical(pd.factorize(trainDf[item])[0])

    trainDf.to_csv(trainFile, index= False)


print("生成提前预测时间为{}的数据集".format(LEAD))
genDataSet(LEAD)

    

    
