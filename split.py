# 按 dimm_sn 号切分 error event 数据集，并保存出现 error 的 dimm 静态信息

import pandas as pd
from config import *
import os, math, json, sys
from multiprocessing import Process, Queue


def saveDfList(dimmMap, dfList):
    
    for sub in dfList:
        sn = sub[0]
        # 若无静态信息，跳过当前DIMM
        if sn not in dimmMap:
            continue
        
        errorDf  = sub[1].reset_index(drop=True)

        subPath = os.path.join(SPLIT_DATA_PATH, sn)
        if not os.path.exists(subPath):
            os.makedirs(subPath)
        
        errorFile = os.path.join(subPath, sn+"_error.csv")
        errorDf.to_csv(errorFile, index=False)
        
        staticFile = os.path.join(subPath, sn+"_static.csv")
        staticDf = pd.json_normalize(dimmMap[sn])
        staticDf.to_csv(staticFile, index=False)
    

    

# 按dimm sn号切分数据，并解析
def splitByDIMM():
    if not os.path.exists(SPLIT_DATA_PATH):
        os.makedirs(SPLIT_DATA_PATH)

    # # 读取CPU信息
    # with open(CPU_INFO, 'r') as json_file:
    #     cpuList = json.load(json_file)
    # cpuMap = {}
    # for cpu in cpuList:
    #     cpuMap[cpu["cpu_sn"]] = cpu
    
    # 读取DIMM静态信息
    
    with open(DIMM_INFO, 'r') as json_file:
        dimmList = json.load(json_file)
    dimmMap = {}
    for dimm in dimmList:
        dimmMap[dimm["dimm_sn"]] = dimm
        
    # 读取error信息
    with open(ERR_INFO, 'r') as json_file:
        data = json.load(json_file)
    df = pd.json_normalize(data)
    df = df.rename(columns={'phy_addr.bank':'bank' ,'phy_addr.bankgroup': 'bankgroup','phy_addr.column':'column' ,
                            'phy_addr.device':'device' ,'phy_addr.rank':'rank' ,'phy_addr.row':'row',
                            'phy_addr.dev':'dev', 'phy_addr.subrank':'subrank'})
    
    df = df.rename(columns={'registers.IA32_MCi_ADDR':'IA32_MCi_ADDR','registers.IA32_MCi_MISC':'IA32_MCi_MISC',
                            'registers.IA32_MCi_STATUS':'IA32_MCi_STATUS','registers.retry_rd_err_log':'retry_rd_err_log',
                            'registers.retry_rd_err_log_address1':'retry_rd_err_log_address1',
                            'registers.retry_rd_err_log_address2':'retry_rd_err_log_address2',
                            'registers.retry_rd_err_log_misc':'retry_rd_err_log_misc',
                            'registers.retry_rd_err_log_parity':'retry_rd_err_log_parity'})
    dfList = list(df.groupby('dimm_sn'))
    
    
    processList = []
    cpuCount = os.cpu_count() * 2
    subListSize = math.ceil(len(dfList) / cpuCount)
    for i in range(cpuCount):
        subDimm = dfList[i*subListSize:(i + 1)*subListSize]
        processList.append(Process(target=saveDfList, args=([dimmMap, subDimm])))
        
    for p in processList:
        p.start()

    for p in processList:
        p.join()
     
[CPU_INFO, DIMM_INFO, ERR_INFO] = sys.argv[1:4]
splitByDIMM()
    

