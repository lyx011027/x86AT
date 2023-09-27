import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Process, Queue
import os, math, json, sys
from config import *


# 多进程处理框架
def multiProcess(dataList, processFunction, processArgs, mergeFunction, mergeArgs, endFlag):
    q = Queue()
    processList = []
    cpuCount = os.cpu_count() * 2
    subListSize = math.ceil(len(dataList) / cpuCount)
    for i in range(cpuCount):
        subDimm = dataList[i*subListSize:(i + 1)*subListSize]
        processList.append(Process(target=processFunction, args=[q, subDimm] + processArgs))
        
    pMerge = Process(target=mergeFunction, args=[q] + mergeArgs)
    pMerge.start()
    for p in processList:
        p.start()

    for p in processList:
        p.join()
    q.put(endFlag)
    pMerge.join()
    
# count 1
def countPredictableUEDIMMMerge(q, leadTimeList, LeadCECountList):
    UE = 0
    timePredictableDIMM = (len(leadTimeList)) * [0]
    countPredictableDIMM = (len(LeadCECountList)) * [0]
    while True:
        LeadTimePredictable,  LeadCountPredictable= q.get()
        if len(LeadTimePredictable) == 0:
            print("UE DIMM count = {}".format(UE))
            for i in range(len(leadTimeList)):
                print("lead time = {}, predictable UE DIMM = {}".format(leadTimeList[i], timePredictableDIMM[i]))
            for i in range(len(LeadCECountList)):
                print("lead count = {}, predictable UE DIMM = {}".format(LeadCECountList[i], countPredictableDIMM[i]))
            return
        UE += 1
        for i in range(len(LeadTimePredictable)):
            if timePredictableDIMM[i]:
                timePredictableDIMM[i] += 1
        for i in range(len(LeadCountPredictable)):
            if countPredictableDIMM[i]:
                countPredictableDIMM[i] += 1

def countPredictableUEDIMMProcess(q, subDfList, leadTimeList, LeadCECountList):
    for item in subDfList:
        LeadTimePredictable = (len(leadTimeList)) * [False]
        LeadCountPredictable = (len(LeadCECountList)) * [False]
        sn = item[0]
        df = item[1]
        
        UERDf = df[df['err_type'].isin(UETypeList)].reset_index(drop=True)
        if UERDf.shape[0] == 0:
            continue
        firstUER = UERDf.loc[0,'record_date']
        
        
        CEDf = df[df['err_type'].isin(CETypeList) & (df['record_date'] < firstUER)].reset_index(drop=True)
        if CEDf.shape[0] > 0:
            timedelta = firstUER - CEDf.loc[0, 'record_date']
            for i in range(len(leadTimeList)):
                if  timedelta > leadTimeList[i]:
                    LeadTimePredictable[i] = True
                else:
                    break
            CECount = CEDf.shape[0]
            for i in range(len(LeadCECountList)):
                if CECount > LeadCECountList[i]:
                    LeadCountPredictable[i] = True
                else:
                    break
            
        q.put(LeadTimePredictable, LeadCountPredictable)
        
# 统计不同 lead time 时，可预测的 UE DIMM 数量
def countPredictableUEDIMM(leadTimeList,LeadCECountList,subDfList):
    multiProcess(subDfList, countPredictableUEDIMMProcess,[leadTimeList, LeadCECountList], countPredictableUEDIMMMerge, [leadTimeList, LeadCECountList], [[],[]])
    

# count 2

# 获取DIMM数量
def getDIMMNum(DIMMDf):

    print("DIMM number = {}".format(DIMMDf.drop_duplicates(subset=["dimm_sn"]).shape[0]))
    dfList = list(DIMMDf.groupby(by=['vendor',"bit_width_x","capacity", "rank_count","speed"]))
    for item in dfList:
        print("vendor = {}, DQ = {}, capacity = {}, rank num = {}, speed = {}, number = {}".format
              ( item[0][0], item[0][1], item[0][2], item[0][3], item[0][4],item[1].shape[0]))
        
# 统计 log 持续时间
def getTime(ErrorDf):
    maxIdx= ErrorDf['record_date'].idxmax()
    minIdx= ErrorDf['record_date'].idxmin()
    
    firstTime = ErrorDf.loc[minIdx, 'record_date']
    lastTime = ErrorDf.loc[maxIdx, 'record_date']
    
    print("first error time : {}, last error time : {}, period : {}".format(firstTime, lastTime, lastTime - firstTime))

# 统计发生不同 type error 的 DIMM 数量
def getErrorTypeDIMMNum(ErrorDf):
    
    errorTypeDf = ErrorDf.drop_duplicates(subset=["err_type"]).reset_index(drop=True)
    errorTypeList = []
    for i in range(errorTypeDf.shape[0]):
        errorTypeList.append(errorTypeDf.loc[i, 'err_type'])
    for errorType in errorTypeList:
        subDf = ErrorDf[ErrorDf['err_type'] == errorType].drop_duplicates(subset=["dimm_sn"]).reset_index(drop=True)
        num = subDf.shape[0]
        print("{} DIMM count = {}".format(errorType, num))
        

# 统计非突发UER中, 首次UER前首次CE与末次CE发生时间

def avg(timeList):
    base = 0
    for time in timeList:
        base += time.days
    return base/len(timeList)

def countBeforeUEMerge(q):
    firstTimeList = []
    lastTimeList = []
    countList = []
    while True:
        tmp = q.get()
        op = tmp[0] 
        if op == -1:
            for i in range(len(firstTimeList)):
                firstTimeList[i] = firstTimeList[i].days * 24* 60*60 + firstTimeList[i].seconds
                lastTimeList[i] = lastTimeList[i].days * 24* 60*60 + lastTimeList[i].seconds
            print("timing gap list of first CE = {}\n".format(firstTimeList))
            print("timing gap list of last CE = {}\n".format(lastTimeList))
            print("CE count before first UE = {}\n".format(countList))
            if len(firstTimeList) > 0 and len(lastTimeList) > 0:
                print("average timing gap of first CE = {}, average timing gap of last CE = {}".
                      format(sum(firstTimeList)/len(firstTimeList), sum(lastTimeList)/len(lastTimeList)))

            return
        if op > 1:
            firstTimeList.append(tmp[1])
            lastTimeList.append(tmp[2])
            countList.append(tmp[3])

def countBeforeUEProcess(q, subDfList):
    
    for item in subDfList:
        sn = item[0]
        df = item[1]
        # df['record_date'] = pd.to_datetime(df['record_date'], format="%Y-%m-%d %H:%M:%S")
        UERDf = df[df['err_type'].isin(UETypeList)].reset_index(drop=True)
        if UERDf.shape[0] == 0:
            continue
        
        firstUER = UERDf.loc[0,'record_date']
        
        CEDf = df[(df['err_type'].isin(CETypeList)) & (df['record_date'] < firstUER) ].reset_index(drop=True)
        if CEDf.shape[0] == 0:
            q.put([1, datetime.now(), datetime.now()])
            continue
        firstCETime = firstUER - CEDf.loc[0, 'record_date'] 
        lastCETime = firstUER - CEDf.loc[CEDf.shape[0] - 1, 'record_date']
        count = CEDf.shape[0]
        q.put([2, firstCETime, lastCETime, count])

def countBeforeUE(dfList):
    multiProcess(dfList, countBeforeUEProcess,[], countBeforeUEMerge, [], [-1,-1,-1,0])


def main():
    [CPU_INFO, DIMM_INFO, ERR_INFO] = sys.argv[1:4]
    
    # 读取DIMM信息
    with open(DIMM_INFO, 'r') as json_file:
        data = json.load(json_file)
    DIMMDf = pd.json_normalize(data)
    
    
    # 读取error信息
    with open(ERR_INFO, 'r') as json_file:
        data = json.load(json_file)
    ErrorDf = pd.json_normalize(data)
    ErrorDf['record_date'] = pd.to_datetime(ErrorDf['record_date'], format="%Y-%m-%d %H:%M:%S")
    ErrorDf = ErrorDf.sort_values(by = ['record_date']).reset_index(drop=True)
    
    ErrorDf = ErrorDf.rename(columns={'phy_addr.bank':'bank' ,'phy_addr.bankgroup': 'bankgroup','phy_addr.column':'column' ,
                            'phy_addr.device':'device' ,'phy_addr.rank':'rank' ,'phy_addr.row':'row',
                            'phy_addr.dev':'dev', 'phy_addr.subrank':'subrank'})
    
    ErrorDf = ErrorDf.rename(columns={'registers.IA32_MCi_ADDR':'IA32_MCi_ADDR','registers.IA32_MCi_MISC':'IA32_MCi_MISC',
                            'registers.IA32_MCi_STATUS':'IA32_MCi_STATUS','registers.retry_rd_err_log':'retry_rd_err_log',
                            'registers.retry_rd_err_log_address1':'retry_rd_err_log_address1',
                            'registers.retry_rd_err_log_address2':'retry_rd_err_log_address2',
                            'registers.retry_rd_err_log_misc':'retry_rd_err_log_misc',
                            'registers.retry_rd_err_log_parity':'retry_rd_err_log_parity'})
    
    ErrorDfList = list(ErrorDf.groupby('dimm_sn'))
    
    print("- count predictable UE DIMM")
    leadTimeList = [timedelta(seconds=0),timedelta(seconds=5),timedelta(minutes=1),timedelta(minutes=5),timedelta(minutes=15),timedelta(minutes=30), timedelta(hours=1)]
    LeadCECountList = [0,1,5,10,20,50,100]
    countPredictableUEDIMM(leadTimeList, LeadCECountList, ErrorDfList)
    print("\n")
    
    print("- count DIMM info")
    getDIMMNum(DIMMDf)
    print('\n')

    print("- count error logging period")
    getTime(ErrorDf)
    print('\n')
    
    print("- count different error type")
    getErrorTypeDIMMNum(ErrorDf)
    print("\n")
    
    print("- count timing gap")
    countBeforeUE(ErrorDfList)
    print("\n")
    
main()