
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "-1";

# Hide messy TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')   #Hide messy Numpy warnings
from pprint import pprint
from DNN.PrepareValues import PrepareDefaultVariablesAndValues
from DNN.NNtrain import NNtrain

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 200)



class Configs:

    startDay = "2018-03-01"
    endDay = "2019-03-01"
    defPath = "/home/features"

    key = "_DNN_NG_Test"

    instrument = 'NG'
    pathLabel = "Ind1"
    target = 'target1'
    test2train = 0.25
    interleaveSplit = False
    trainLabel = "test"

    batchTraining = False
    batchNormalization = False
    saveMeansStd = True
    predictionTask = "classification"                                     ### Classification or regression task
    maxEvals = 3
    earlyStop = 10
    earlyStop_minDelta = 0.0001

    fImportanceFile = './DNN/ImportanceGrad_0.01_2018-03-01_2019-04-05.csv'
    useTopImportantFeatures = "all"
    useTensorBoard = False

    logLevel = 2                                                          ### If logLevel=2 --> no full training history shows. Only at the end of epoch
    batchTrMaxBS = 1
    clth = 0                                                              ### if 0 rescale dq target 1 if >0 else 0

    hpSpace = {
        'epochs': [10000],
        'hu': [100, 150, 200, 250, 300, 350, 400],
        'hu2': [0],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'act1': ['sigmoid'],
        'act2': ['sigmoid'],
        'optimizer': ['adam'],
        'loss': ['binary_crossentropy'],
        'bSize': [50000]                                                  ### If BatchTrain batchSize = 1 day
    }




print("\nConfigs:")
pprint(vars(Configs))

st = datetime.now()
defaultVariables = PrepareDefaultVariablesAndValues(Configs=Configs)

dd = NNtrain(defaultVariables=defaultVariables, Configs=Configs)
dd.train()
summary = dd.getResult()

print(summary)
print("Total Time: " + str(datetime.now() - st))



