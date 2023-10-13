from config import *
import pandas as pd
def getLSTMItem():
    sample = {}
    # sample = getFrequencySample(sample)

    sample = getSubBankSample(sample)
    sample = getCECountSample(sample)

    
    return list(sample.keys())
soureFile = os.path.join(DATA_SET_PATH, dataSetFile)

df = pd.read_csv(soureFile, low_memory=False)

df = df[getLSTMItem()+['dimm_sn','label']]

df.to_csv(os.path.join(dataSetFile), index=False)
