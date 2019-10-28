
import os
import json
import re

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score
from datetime import datetime

from hyperopt import STATUS_OK, Trials, fmin, tpe, hp

from DNN.Batch_Generator import Batch_Generator

import keras
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import DNN.lossFunctions as lossFunctions




print("\nGPU Available: " + str(tf.test.is_gpu_available()) + "\n")



def l1group(hu, kappa):
    def l1(M):
        return kappa*K.sum(np.sqrt(hu)*K.sqrt(K.sum(K.square(M), axis = 1)))
    return l1


class NNtrain(object):
    def  __init__(self, defaultVariables, Configs):


        self.Configs = Configs
        self.defaultVariables = defaultVariables
        self.modelsResults = pd.DataFrame()

        self.readyToTrain = True
        self.modelsPath = "./Models%s/" % (self.Configs.key)
        if not os.path.exists(self.modelsPath):
            os.makedirs(self.modelsPath)
        else:
            print("\n\nWarning! " + self.modelsPath + " Already exists. Use another key.\n\n")

        self.tensorboardLogPath = "./TensorBoardLogs/"
        if not os.path.exists(self.tensorboardLogPath):
            os.makedirs(self.tensorboardLogPath)

        ### Save Means, Std
        if self.Configs.saveMeansStd:
            self.defaultVariables.saveMeansStd(modelsPath=self.modelsPath)
        ### Save train x, y files names
        self.defaultVariables.saveTrainTestDates(modelsPath=self.modelsPath)

        self.runStartTime = datetime.now()
        self.fImportance = []
        self.fImportanceIndex = np.arange(0, len(self.defaultVariables.indicatorsСolumns), 1)

        self.initDefaultVariables()
        self.initSpaces()

        self.runAutoSaveThread = True
        self.trialsResults = []
        self.trialsResultsLen = 0
        self.csvSummaryFileName = self.modelsPath + "/ModelsResults_" + str(re.sub('[:.]', "", str(self.runStartTime).split(" ")[-1]))[:6] + "_" + self.trainLabel + ".csv"
        self.writeConfigToLog()



    def writeConfigToLog(self):
        configFileName = self.modelsPath + "/Configs_" +  str(re.sub('[:.]', "", str(self.runStartTime).split(" ")[-1]))[:6] + "_"  + self.trainLabel + ".cfg"
        dd = dict(self.Configs.__dict__)
        with open(configFileName, 'w') as f:
            f.write('Configs: \n\n')
            for key, value in dd.items():
                if bool(re.findall(r"^[a-zA-Z0-9]+$", key)):
                    #print(key, value)
                    f.write('   %s : %s\n' % (key, value))


    def initDefaultVariables(self):

        self.defaultVariables.key = self.Configs.key
        self.instrument = self.Configs.instrument
        self.trainLabel = self.Configs.trainLabel
        self.target = self.Configs.target
        self.batchTraining = self.Configs.batchTraining
        self.batchNormalization = self.Configs.batchNormalization
        self.predictionTask = self.Configs.predictionTask
        self.maxEvals = self.Configs.maxEvals
        self.earlyStop = self.Configs.earlyStop
        self.logLevel = self.Configs.logLevel
        self.batchTrMaxBS = self.Configs.batchTrMaxBS

        self.IndMeans = self.defaultVariables.IndMeans
        self.IndStds = self.defaultVariables.IndStds

        self.hpSpace = self.Configs.hpSpace

        self.targetsСolumns = self.defaultVariables.targetsСolumns
        self.indicatorsСolumns = self.defaultVariables.indicatorsСolumns

        self.X_TrainFiles = self.defaultVariables.X_TrainFiles
        self.X_TestFiles = self.defaultVariables.X_TestFiles
        self.Y_TrainFiles = self.defaultVariables.Y_TrainFiles
        self.Y_TestFiles = self.defaultVariables.Y_TestFiles

        self.yTargetIndex = self.targetsСolumns.index(self.Configs.target)

        self.filterFeaturesImportance()

        self.X1 = self.defaultVariables.X1[:, self.fImportanceIndex]
        self.Y1 = self.defaultVariables.Y1[:, self.yTargetIndex ]
        self.X2 = self.defaultVariables.X2[:, self.fImportanceIndex]
        self.Y2 = self.defaultVariables.Y2[:, self.yTargetIndex ]
        self.Y_True = self.defaultVariables.Y_True[:, self.yTargetIndex ]

        print("\nX1 shape: " + str(self.X1.shape))
        print("X2 shape: " + str(self.X2.shape))


    def initSpaces(self):
        if self.batchTraining:
            ### For batch train. Check if in config are batch sizes greater batchTrMaxBS.
            maxthreshold = self.batchTrMaxBS
            self.hpSpace["bSize"] = [x if x<maxthreshold else maxthreshold for x in self.hpSpace["bSize"]]
        self.space = {key: hp.choice(key, self.hpSpace[key]) for key in self.hpSpace}


    def filterFeaturesImportance(self):
        if self.Configs.useTopImportantFeatures == "all":
            self.fImportance = []
            self.fImportanceIndex = np.arange(0, len(self.indicatorsСolumns), 1)
        else:
            if len(self.defaultVariables.featuresImportance)>0:
                fImpTemp = self.defaultVariables.featuresImportance.sort_values("Importance", ascending=False)
                fImpTemp = fImpTemp.head(int(self.Configs.useTopImportantFeatures))

                self.fImportance = fImpTemp.Feature.tolist()
                self.indicatorsСolumns = self.fImportance

                self.fImportanceIndex = [self.defaultVariables.indicatorsСolumns.index(x) for x in self.fImportance]

                print("\nSelected top %s features." % len(self.fImportance))
                print("With total importance: %.3f / %.2f." % (fImpTemp.Importance.sum(), self.defaultVariables.featuresImportance.Importance.sum()))
                print("Filtered features ids: %s \n" % self.fImportanceIndex)
            else:
                self.fImportance = []
                self.fImportanceIndex = np.arange(0, len(self.indicatorsСolumns), 1)



    def hyperParametersOptimizationBT(self, X1, X2, Y1, Y2):
        print("HyperParameters Optimization")

        self.trials = Trials()
        st = datetime.now()
        fmin(lambda args: self.objectiveBatchTraining(args=args, X1=X1, X2=X2, Y1=Y1, Y2=Y2, Y_True=self.Y_True, means=self.IndMeans, stds=self.IndStds),
             space=self.space, algo=tpe.suggest, max_evals=self.maxEvals, trials=self.trials)
        print("\nPart model training took: " +  str(datetime.now()-st) + "\n")


    def hyperParametersOptimizationFT(self):
        print("HyperParameters Optimization")

        self.trials = Trials()
        st = datetime.now()
        fmin(lambda args: self.objectiveFullTraining(args=args, X1=self.X1, X2=self.X2, Y1=self.Y1, Y2=self.Y2),
             space=self.space, algo=tpe.suggest, max_evals=self.maxEvals ,trials=self.trials)
        print("\nPart model training took: " +  str(datetime.now()-st) + "\n")



    @staticmethod
    def formResult(score, trialId, hu, hu2, dropout,act1, act2, optimizer, epochs, epochsUsed, loss, batch_size, batchNormalization, min_loss, modelName, status):
        res = {
            'loss': -score,
            'info': {'id': trialId, 'hu': hu, 'hu2': hu2, 'dropout': dropout, 'act1': act1, 'act2': act2,
                      'optimizer': optimizer, 'maxEpoch':epochs, 'epochsUsed': epochsUsed,
                      'loss': loss, 'bSize': batch_size, 'batchNormalization': batchNormalization,
                      'min_loss': min_loss, 'score': score, 'modName': modelName
                      },
            'status': status
        }
        return res


    def objectiveBatchTraining(self, args, X1, X2, Y1, Y2, Y_True, means, stds):

        epochs = args["epochs"]
        hu = args['hu']
        hu2 = args['hu2']
        dropout = args['dropout']
        act1 = args['act1']
        act2 = args['act2']
        optimizer = args['optimizer']
        loss = args['loss']
        batch_size = args['bSize']
        nSamples = len(X1) // batch_size

        trialId = self.trials.trials[-1]["tid"]

        print("\nStart batch %s model training.")
        print("TrialId=%s Instrument=%s target=%s hu=%s dropout=%s maxEpochs=%s batchSize=%s nSamples=%s\n" %
              (trialId, self.instrument, self.target, hu, dropout, epochs, batch_size, nSamples))

        trainingGenerator = Batch_Generator(X1, Y1, batch_size, yTargetIndex=self.yTargetIndex, batchNormalization=self.batchNormalization,
                                            volumeLeftCol=self.volumeLeftCol, means=means, stds=stds)
        validationGenerator = Batch_Generator(X2, Y2, batch_size, yTargetIndex=self.yTargetIndex, batchNormalization=self.batchNormalization,
                                              volumeLeftCol=self.volumeLeftCol, means=means, stds=stds)

        steps_per_epoch = len(X1) // batch_size
        validation_steps = len(X2) // batch_size

        modelName = self.modelsPath + 'model_BT_{}_{}_{}_{}.hdf5'.format(self.instrument, self.target, trialId, self.trainLabel)
        tbLogsPath = self.tensorboardLogPath + self.modelsPath.split("/")[1] + "_{}_{}".format(self.instrument, trialId)
        tensorboard = TensorBoard(log_dir=tbLogsPath, histogram_freq=0, write_graph=True, write_images=True)

        if loss in dir(lossFunctions):

            dnnInput = Input(shape=(len(self.indicatorsСolumns),), name='dnn_input')

            if self.batchNormalization:
                dnnModel = Dense(hu, use_bias=False, activation=act1,  name="Dense1_BN")(dnnInput)
                dnnModel = BatchNormalization()(dnnModel)
            else:
                dnnModel = Dense(hu, use_bias=True, activation=act1,  name="Dense1_FN")(dnnInput)
            dnnModel = Dropout(dropout)(dnnModel)

            if hu2>0:
                dnnModel = Dense(hu2, use_bias=False, activation=act1,  name="Dense2")(dnnModel)
                dnnModel = Dropout(dropout)(dnnModel)

            dnnOut = Dense(1, activation=act2, name="Dense_Output")(dnnModel)

            model = Model(inputs=[dnnInput], outputs=dnnOut)
            model.compile(loss=getattr(lossFunctions, loss), optimizer=optimizer, metrics=['accuracy'])

            if self.Configs.useTensorBoard:
                history = model.fit_generator(generator=trainingGenerator,
                                              steps_per_epoch=steps_per_epoch,
                                              epochs=epochs,
                                              verbose=self.logLevel,
                                              validation_data=validationGenerator,
                                              validation_steps=validation_steps,
                                              callbacks=[
                                                  ModelCheckpoint(modelName, save_best_only=True),
                                                  EarlyStopping(monitor='val_loss', mode='min', patience=self.earlyStop, verbose=self.logLevel),
                                                  tensorboard
                                              ])
            else:
                history = model.fit_generator(generator=trainingGenerator,
                                              steps_per_epoch=steps_per_epoch,
                                              epochs=epochs,
                                              verbose=self.logLevel,
                                              validation_data=validationGenerator,
                                              validation_steps=validation_steps,
                                              callbacks=[
                                                  ModelCheckpoint(modelName, save_best_only=True),
                                                  EarlyStopping(monitor='val_loss', mode='min', patience=self.earlyStop, verbose=self.logLevel)
                                              ])

            ### Save models training history
            history_dict = history.history
            historyPath = self.modelsPath + 'modelHistory_BT_{}_{}_{}_{}.hdf5'.format(self.instrument, self.target, trialId, self.trainLabel)
            json.dump(history_dict, open(historyPath, 'w'))

            ### Load best model for validation
            model = load_model(self.modelsPath + 'model_FT_{}_{}_{}_{}.hdf5'.format(self.instrument, self.target, trialId, self.trainLabel),
                               custom_objects={loss: getattr(lossFunctions, loss)})
            pred = model.predict_generator(validationGenerator, validation_steps).ravel()

            if self.predictionTask=="classification":
                score = roc_auc_score(Y_True, pred)
            elif self.predictionTask=="regression":
                score = roc_auc_score(Y_True, pred)
            else:
                score = 0

            min_loss = min(history.history['val_loss'])
            epochsUsed = len(history_dict["val_loss"])

            res = self.formResult(score, trialId, hu, hu2, dropout,act1, act2, optimizer, epochs, epochsUsed, loss, batch_size, self.batchNormalization, min_loss, modelName, STATUS_OK)
        else:
            print("Warning! Specified loss function: '" + loss + "' is not found in lossFunctions.py.\n")
            res = self.formResult(0, trialId, hu, hu2, dropout, act1, act2, optimizer, epochs, 0, loss,
                                  batch_size, self.batchNormalization, 0, modelName, STATUS_OK)

        self.modelsResults = self.modelsResults.append(pd.DataFrame([res["info"]]))
        self.saveResults()
        keras.backend.clear_session()
        if self.logLevel == 0:
            print('Score: {:.3f}, loss: {:.3f}'.format(score, min_loss))
        return res


    def objectiveFullTraining(self, args, X1, X2, Y1, Y2):

        epochs = args["epochs"]
        hu = args['hu']
        hu2 = args['hu2']
        dropout = args['dropout']
        act1 = args['act1']
        act2 = args['act2']
        optimizer = args['optimizer']
        loss = args['loss']

        trialId = self.trials.trials[-1]["tid"]
        batch_size = args['bSize']
        nSamples = X1.shape[0] // batch_size

        print("\nStart full model training.")
        print("TrialId=%s Instrument=%s target=%s hu=%s dropout=%s maxEpochs=%s batchSize=%s nSamples=%s\n" %
              (trialId, self.instrument, self.target, hu, dropout, epochs, batch_size, nSamples))

        modelName = self.modelsPath + 'model_FT_{}_{}_{}_{}.hdf5'.format(self.instrument, self.target, trialId, self.trainLabel)
        tbLogsPath = self.tensorboardLogPath + self.modelsPath.split("/")[1] + "_{}_{}".format(self.instrument, trialId)
        tensorboard = TensorBoard(log_dir=tbLogsPath, histogram_freq=0, write_graph=True, write_images=True)

        if loss in dir(lossFunctions):

            dnnInput = Input(shape=(len(self.indicatorsСolumns),), name='dnn_input')

            if self.batchNormalization:
                dnnModel = Dense(hu, use_bias=False, activation=act1,  name="Dense1_BN")(dnnInput)
                dnnModel = BatchNormalization()(dnnModel)
            else:
                dnnModel = Dense(hu, use_bias=True, activation=act1,  name="Dense1_FN")(dnnInput)
            dnnModel = Dropout(dropout)(dnnModel)

            if hu2>0:
                dnnModel = Dense(hu2, use_bias=False, activation=act1,  name="Dense2")(dnnModel)
                dnnModel = Dropout(dropout)(dnnModel)

            dnnOut = Dense(1, activation=act2, name="Dense_Output")(dnnModel)

            model = Model(inputs=[dnnInput], outputs=dnnOut)
            model.compile(loss=getattr(lossFunctions, loss), optimizer=optimizer, metrics=['accuracy'])
            if self.Configs.useTensorBoard:
                history = model.fit(X1, Y1, batch_size=batch_size, epochs=epochs, verbose=self.logLevel, validation_data=(X2, Y2),
                                    callbacks=[
                                        ModelCheckpoint(modelName, save_best_only=True),
                                        EarlyStopping(monitor='val_loss', mode='min', patience=self.earlyStop, verbose=self.logLevel),
                                        tensorboard
                                    ])
            else:
                history = model.fit(X1, Y1, batch_size=batch_size, epochs=epochs, verbose=self.logLevel, validation_data=(X2, Y2),
                                    callbacks=[
                                        ModelCheckpoint(modelName, save_best_only=True),
                                        EarlyStopping(monitor='val_loss', mode='min', patience=self.earlyStop, verbose=self.logLevel)
                                    ])

            ### Save models training history
            history_dict = history.history
            historyPath = self.modelsPath + 'modelHistory_FT_{}_{}_{}_{}.hdf5'.format(self.instrument, self.target, trialId, self.trainLabel)
            json.dump(history_dict, open(historyPath, 'w'))

            ### Load best model for validation
            model = load_model(self.modelsPath + 'model_FT_{}_{}_{}_{}.hdf5'.format(self.instrument, self.target, trialId, self.trainLabel),
                               custom_objects={loss: getattr(lossFunctions, loss)})
            pred = model.predict(X2)

            if self.predictionTask=="classification":
                score = roc_auc_score(Y2, pred)
            elif self.predictionTask=="regression":
                score = roc_auc_score(Y2, pred)
            else:
                score = 0

            min_loss = min(history.history['val_loss'])
            epochsUsed = len(history_dict["val_loss"])

            res = self.formResult(score, trialId, hu, hu2, dropout,act1, act2, optimizer, epochs, epochsUsed, loss, batch_size, self.batchNormalization, min_loss, modelName, STATUS_OK)
        else:
            print("Warning! Specified loss function: '" + loss + "' is not found in lossFunctions.py.\n")
            res = self.formResult(0, trialId, hu, hu2, dropout, act1, act2, optimizer, epochs, 0, loss,
                                  batch_size, self.batchNormalization, 0, modelName, STATUS_OK)

        self.modelsResults = self.modelsResults.append(pd.DataFrame([res["info"]]))
        self.saveResults()
        keras.backend.clear_session()
        if self.logLevel == 0:
            print('Score: {:.3f}, loss: {:.3f}'.format(score, min_loss))
        return res


    def saveResults(self):
        if len(self.modelsResults)>0:
            self.modelsResults = self.modelsResults.sort_values('score', ascending=False)
            self.modelsResults.to_csv(self.csvSummaryFileName, index=False)


    def train(self):

        if self.readyToTrain:
            if self.batchTraining:
                print("Run batch training. Only one bach will be in memory.")
                self.hyperParametersOptimizationBT(X1=self.X_TrainFiles, X2=self.X_TestFiles, Y1=self.Y_TrainFiles, Y2=self.Y_TestFiles)
            else:
                print("Run full training. All data will be loaded into memory.")
                self.hyperParametersOptimizationFT()

            self.runAutoSaveThread = False



    def getResult(self):
        result = self.modelsResults
        result = result.reset_index(drop=True)
        return result
