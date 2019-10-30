
# Neural network training workflow

This workflow allow to train and optimize NN using arbitrary data set.
But data for training shoud be aggregate in correct form.
Path to data controlled from config as follows:
    dataDir = Configs.defPath / Configs.instrument / Configs.pathLabel

dataDir should have for each data separate folder.
For example, in dataDir should be folders: "2019-01-01", "2019-01-02", "2019-01-03" ... etc.
In each of these folders should be two files: indicators.npy - data set with features, targets.npy - data set with targets.


### Requiremets:
* python 3.6
* pandas
* numpy
* keras
* tensorflow
* sklearn

### Config description:

    startDay - Start day for traing in dataDir folder
    endDay - End day for traing in dataDir folder
    defPath - default path to dataDir
    key - label that will be added to results folder name
    instrument - default path to dataDir
    pathLabel - default path to dataDir
    target - target column name
    test2train - test to train split ratio
    interleaveSplit - use interleave test to train split (each three days to train, each 4-th day to test)
    trainLabel - label that will be added to model

    batchTraining - use or not batch training
    batchNormalization - use or not batch normalization
    saveMeansStd - save or not means and standart deviations
    predictionTask - classification or regression task
    maxEvals - number of evaluations
    earlyStop - early stop for NN. If no better results mode than earlyStop epochs, then sstop trainig

    fImportanceFile - feature importance file, if exists.
    useTopImportantFeatures - use numbver of best features in training from fImportanceFile. ("all", number>0).
    useTensorBoard = False

    logLevel - logLevel=2 --> no full training history shows. Only at the end of epoch
    batchTrMaxBS = 1
    clth - if 0 rescale regression target 1 if >0 else 0

    hpSpace = {
        'epochs': [10000],                                        - number of eposhs to use
        'hu': [100, 150, 200, 250, 300, 350, 400],                - number of hidden units in first hidded layer
        'hu2': [0],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], - number of hidden units in second hidded layer
        'act1': ['sigmoid'], 
        'act2': ['sigmoid'],
        'optimizer': ['adam'],
        'loss': ['binary_crossentropy'],
        'bSize': [50000]                                          - if BatchTrain batchSize = 1 day
    }


Check Configs in trainDNN.py
If you want to run multiple targets, check targets list in trainDNN.py before 'for' loop.
If you want to add your own new loss function, you can add it into lossFunctions.py, and then specify it in trainDNN.py Config class.

Run: python trainDNN.py
Run TensorBoard(in directory where trainDNN.py): tensorboard --logdir='TensorBoardLogs/' --host=10.12.1.59 --port=8999  (check if port not busy)


### NN architeture scheme:
![alt text](https://github.com/romario076/Neural-network-train-workflow/blob/master/DNN/scheme.jpg)
