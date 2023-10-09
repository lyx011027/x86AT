import pandas as pd
from datetime import datetime, timedelta
import csv
from config import *
import os
import math
import copy
from multiprocessing import Process, Queue
# import moxing as mox
MAXROW = 30
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
    for item in STATIC_ITEM:
        
        sample[item] = staticDf.loc[0,item]

    
    sample['dimm_sn'] = dimm
    
    return sample


def parseBitError(parity):
    
    if pd.isna(parity):
        return [-1] * 7
    
    adjDqCount = 0
    DQMap = {}
    BurstList = []
    for i in range(8):
        DQ = int(parity[i + 2], 16)
        for j in range(3):
            if (DQ >>j & 3) == 3:
                adjDqCount += 1
                
        for j in range(4):
            if (DQ >>j & 1) == 1:
                DQMap[4-j-1] = True
        
        if DQ > 0:
            BurstList.append(i)
    DQList = sorted(list(DQMap.keys()))
    
    DQCount = len(DQList)
    BurstCount = len(BurstList)
    if DQCount > 1:
        maxDqDistance = DQList[DQCount - 1] - DQList[0]
        minDQDistance = 8
        for i in range(1, DQCount):
            
            minDQDistance = min(minDQDistance, DQList[i] - DQList[i - 1])
    else:
        maxDqDistance = -1
        minDQDistance = -1
        
    if BurstCount > 1:
        maxBurstDistance = BurstList[BurstCount - 1] - BurstList[0]
        minBurstDistance = 8
        for i in range(1, BurstCount):
            minBurstDistance = min(minBurstDistance, BurstList[i] - BurstList[i - 1])
    else:
        maxBurstDistance = -1
        minBurstDistance = -1
    
    
    
   
    
    return [adjDqCount, maxDqDistance, minDQDistance,DQCount, maxBurstDistance, minBurstDistance, BurstCount]
        
# print(parseBitError('0x040000000'))

def addItem(sample, item,List):
    sample['max_{}'.format(item)] = max(List)
    sample['min_{}'.format(item)] = min(List)
    sample['sum_{}'.format(item)]  = sum(List)
    sample['avg_{}'.format(item)]  = sample['sum_{}'.format(item)] / len(List)
    return sample

def addItemExclude(sample, item,List):
    
    List = list(filter(lambda x:x >= 0, List))
    if len(List) > 0:
        sample['max_{}'.format(item)] = max(List)
        sample['min_{}'.format(item)] = min(List)
        sample['sum_{}'.format(item)]  = sum(List)
        sample['avg_{}'.format(item)]  = sample['sum_{}'.format(item)] / len(List)
    return sample

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
            errorTime = error['err_time']
            
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


def processDimm(id, q, dimmList, leadTime):
    for dimm in dimmList:
        # print(dimm)
        
        # 生成静态信息
        staticFile = os.path.join(SPLIT_DATA_PATH, dimm, dimm+"_static.csv")
        baseSample = getBaseSample(dimm,staticFile) 
        errorFile = os.path.join(SPLIT_DATA_PATH, dimm, dimm+"_error.csv")
        df = pd.read_csv(errorFile, low_memory=False)
        df['record_date'] = pd.to_datetime(df['record_date'], format="%Y-%m-%d %H:%M:%S")
        
        UEFlag = False
        firstUER = datetime.now().replace(year=2099)
        UEDf = df[(df['err_type'].isin(UETypeList))].reset_index(drop=True)
        if UEDf.shape[0] != 0:
            UEFlag = True
            firstUER = UEDf.loc[0, 'record_date']
        CEDf = df[(df['record_date'] <  firstUER)].reset_index(drop=True)
        if CEDf.shape[0] == 0:
            continue
        
        # firstCE, lastCE = CEDf.loc[0, 'record_date'], CEDf.loc[CEDf.shape[0] - 1, 'record_date']
        # if UEFlag and (firstUER - firstCE < timedelta(minutes=30) or  firstUER - lastCE > timedelta(days=45)):
        #     continue
        
        CECount =CEDf.shape[0]
        if CECount < sampleDistance:
            continue
        # 故障记录
        sampleList = []

        accumulateCE = 1
        
        
        timeList = []
        adjDqLsit = []
        MaxDqDistanceLsit = []
        MinDqDistanceLsit = []
        MaxBurstDistanceLsit = []
        MinBurstDistanceLsit = []
        BurstCountList = []
        DqCountList = []
        
        bankGroupMap = {}
        CE_number = 0
        centerList = []
        CEList = []
        for  index, error in CEDf.iterrows():
            errorTime = error['record_date']
            

            parity = error['retry_rd_err_log_parity']
            
            flag = False
            for i in range(len(timeList)):
                if errorTime - timeList[i] < OBSERVATION:
                    break
                else:
                    flag = True
            if flag:
                timeList = timeList[i+1:]
                adjDqLsit = adjDqLsit[i+1:]
                MaxDqDistanceLsit = MaxDqDistanceLsit[i+1:]
                MinDqDistanceLsit = MinDqDistanceLsit[i+1:]
                MaxBurstDistanceLsit = MaxBurstDistanceLsit[i+1:]
                MinBurstDistanceLsit = MinBurstDistanceLsit[i+1:]
                BurstCountList = BurstCountList[i+1:]
                DqCountList = DqCountList[i+1:]
            # timeList = timeList[:10]
            # adjDqLsit = adjDqLsit[:10]
            # MaxDqDistanceLsit = MaxDqDistanceLsit[:10]
            # MinDqDistanceLsit = MinDqDistanceLsit[:10]
            # MaxBurstDistanceLsit = MaxBurstDistanceLsit[:10]
            # MinBurstDistanceLsit = MinBurstDistanceLsit[:10]
            # BurstCountList = BurstCountList[:10]
            # DqCountList = DqCountList[:10]
            
            
            adjDqCount, maxDqDistance, minDQDistance,DQCount, maxBurstDistance, minBurstDistance, BurstCount = parseBitError(parity)
            
            timeList.append(errorTime)
            
            adjDqLsit.append(adjDqCount)
            BurstCountList.append(BurstCount)
            DqCountList.append(DQCount)
            
            MaxDqDistanceLsit.append(maxDqDistance)
            MinDqDistanceLsit.append(minDQDistance)
            MaxBurstDistanceLsit.append(maxBurstDistance)
            MinBurstDistanceLsit.append(minBurstDistance)
            
            
            CE_number += 1
            
            if error['with_phy_addr']:
                rowId, columnId, bankId, bankgroupId =  error['row'], error['column'], error['bank'], error['bankgroup']
                if bankgroupId not in bankGroupMap:
                    bankGroupMap[bankgroupId] = {}
                
                if bankId not in bankGroupMap[bankgroupId]:
                    bankGroupMap[bankgroupId][bankId] = {}
                    
                
                position = (rowId,columnId) 
                if position not in bankGroupMap[bankgroupId][bankId]:
                    bankGroupMap[bankgroupId][bankId][position] = 0
                bankGroupMap[bankgroupId][bankId][position] += 1
                # else:
                #     baseSample['noadd'] += 1
                
                centerList = addCenter(centerList, error)
            
            
            
            # if index != CEDf.shape[0]-1 and errorTime - lastCE < Interval:
            # if index != CEDf.shape[0]-1 and accumulateCE < onceCount:
            if  accumulateCE < onceCount:
                accumulateCE += 1
                continue
            accumulateCE = 1
            
            lastCE = errorTime
            
            sample = copy.copy(baseSample)
            sample['time'] = errorTime.timestamp()
             
            sample['label'] = firstUER - errorTime < Predict
            sample = addItemExclude(sample, 'adjERRDq',adjDqLsit)
            sample = addItemExclude(sample, 'errorDq',DqCountList)
            sample = addItemExclude(sample, 'errorBurst',BurstCountList)
            
            sample = addItemExclude(sample, 'maxDqDistance',MaxDqDistanceLsit)
            sample = addItemExclude(sample, 'minDqDistance',MinDqDistanceLsit)
            sample = addItemExclude(sample, 'maxBurstDistance',MaxBurstDistanceLsit)
            sample = addItemExclude(sample, 'minBurstDistance',MinBurstDistanceLsit)
            
            sample = addSubBankSample(sample, centerList, CEList)
            
            sample = addCECount(sample, bankGroupMap,CE_number)
            
            sampleList.append(sample)
            
        if UEFlag == 1:
            print(dimm)
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
    cpuCount = os.cpu_count()
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

    

    