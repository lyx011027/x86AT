# 统计相同位置是否会有不同的bit error
import pandas as pd
import pandas as pd
from datetime import datetime, timedelta
import csv
from config import *
import os
import math
import copy
from multiprocessing import Process, Queue

def parseBitError(parity):
    
    if pd.isna(parity):
        return -1
    
    DQMap = {}
    for i in range(8):
        DQ = int(parity[i + 2], 16)

                
        for j in range(4):
            if (DQ >>j & 1) == 1:
                DQMap[4-j-1] = True
    DQList = sorted(list(DQMap.keys()))
    
    DQCount = len(DQList)
    if DQCount > 1:
        maxDqDistance = DQList[DQCount - 1] - DQList[0]
        minDQDistance = maxDqDistance
        for i in range(1, DQCount):
            
            minDQDistance = min(minDQDistance, DQList[i] - DQList[i - 1])
    else:
        maxDqDistance = -1
        minDQDistance = -1
    return minDQDistance

def processDimm(id, q, dimmList):
    for dimm in dimmList:
        # print(dimm)
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
        
        positionMap = {}
        
        for _, error in CEDf.iterrows():
            position = (error['column'], error['bankgroup'], error['bank'], error['rank'], error['row'])
            retry_rd_err_log_parity = error['retry_rd_err_log_parity']
            if position not in positionMap:
                positionMap[position] = {}
            positionMap[position][retry_rd_err_log_parity] = True
        flag = False
        for position in positionMap.keys():
            count = 0
            
            for bit in positionMap[position].keys():
                if parseBitError(bit) == 1:
                    count += 1
            if count >= 1:        
                flag = True
        if UEFlag:
            if flag :
                q.put(0)
            else:
                q.put(1)
        else:
            if flag :
                q.put(2)
            else:
                q.put(3)
            
            
def mergeFunction(q):
    L = [0,0,0,0]
    while True:
        flag= q.get()
        if  flag == -1:
            print(L)
            precision = (L[0] /(L[0] + L[2]))
            recall = (L[0] /(L[0] + L[1]))
            print("precision:{}, recall:{}".format(precision, recall))
            break
        L[flag] += 1
def genDataSet():
    
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
        processList.append(Process(target=processDimm, args=(i,q, subDimm)))
        



    pMerge = Process(target=mergeFunction, args=[q])
    pMerge.start()
    for p in processList:
        p.start()

    for p in processList:
        p.join()
    q.put(-1)
    pMerge.join()

genDataSet()

    

    
