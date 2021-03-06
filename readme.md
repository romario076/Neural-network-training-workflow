
# Neural network training workflow

This workflow allow to train and optimize NN using arbitrary data set for regression and binary classification problem.<br>
But data for training shoud be aggregate in correct form.<br>

Path to data controlled from config as follows:<br>
>dataDir = Configs.defPath / Configs.instrument / Configs.pathLabel

For each Date shoud exists separate folder in 'dataDir' directory.<br>
For example, in dataDir should be folders: "2019-01-01", "2019-01-02", "2019-01-03" ... etc.<br>
In each of these folders should be two files: 
* indicators.npy - data set with features
* targets.npy - data set with targets.


There are two ways to train the model, depending on the size of the input data set.<br>
If there is a lot of free RAM, then you can load all the data into memory. (*batchTraining = False*)<br>
If the amount of training data is very large, you can use the option to train in parts. (*batchTraining = True*)<br>
If batchTraining = True, then for one batch will be taken one day, or one file.<br>


### Requiremets:
* python 3.6
* pandas
* numpy
* keras
* tensorflow
* sklearn

<hr>

### Config description:
 * **startDay** - Start day for traing in dataDir folder
 * **endDay** - End day for traing in dataDir folder
 * **defPath** - default path to dataDir
 * **key** - label that will be added to results folder name
 * **instrument** - default path to dataDir
 * **pathLabel** - default path to dataDir
 * **target** - target column name
 * **test2train** - test to train split ratio
 * **interleaveSplit** - use interleave test to train split (each three days to train, each 4-th day to test)
 * **trainLabel** - label that will be added to model

 * **batchTraining** - use or not batch training
 * **batchNormalization** - use or not batch normalization
 * **saveMeansStd** - save or not means and standart deviations
 * **predictionTask** - classification or regression task
 * **maxEvals** - number of evaluations
 * **earlyStop** - early stop for NN. If no better results mode than earlyStop epochs, then sstop trainig
 * **earlyStop_minDelta** - when the improvement(prev-current) higher than this delta it is considered as improvement

 * **fImportanceFile** - feature importance file, if exists.
 * **useTopImportantFeatures** - use numbver of best features in training from fImportanceFile. ("all", number>0).
 * **useTensorBoard** = False

 * **logLevel** - logLevel=2 --> no full training history shows. Only at the end of epoch
 * **batchTrMaxBS** = 1
 * **clth** - if 0 rescale regression target 1 if >0 else 0

 * **hpSpace** = { <br>
        'epochs': [10000],                                        - number of eposhs to use<br> 
        'hu': [100, 150, 200, 250, 300, 350, 400],                - number of hidden units in first hidded layer<br> 
        'hu2': [0],<br> 
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], - number of hidden units in second hidded layer<br> 
        'act1': ['sigmoid'], <br> 
        'act2': ['sigmoid'],<br> 
        'optimizer': ['adam'],<br> 
        'loss': ['binary_crossentropy'],<br> 
        'bSize': [50000]                                          - if BatchTrain batchSize = 1 day<br> 
    }<br> 
    
<hr>

Check Configs in trainDNN.py<br>
If you want to run multiple targets, check targets list in trainDNN.py before 'for' loop.<br>
If you want to add your own new loss function, you can add it into lossFunctions.py, and then specify it in trainDNN.py Config class.<br>

**Run training**: python trainDNN.py<br>
**Run TensorBoard**(in directory where trainDNN.py): tensorboard --logdir='TensorBoardLogs/' --host=10.12.1.59 --port=8999  (check if port not busy)<br>

<hr>

### NN architeture scheme:
![alt text](https://github.com/romario076/Neural-network-train-workflow/blob/master/DNN/scheme.jpg)

<hr>

### Tensorboard output during training:
![alt text](https://user-images.githubusercontent.com/10981310/68022506-2e6dcb80-fcad-11e9-8458-848858ab1871.png)

<hr>
