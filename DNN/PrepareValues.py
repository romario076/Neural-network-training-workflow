
import os
import re
import numpy as np
from datetime import datetime
import pandas as pd




class PrepareDefaultVariablesAndValues(object):
    def __init__(self, Configs):
        print("\nPrepare default variables and values.")

        self.Configs = Configs

        self.batchTraining = self.Configs.batchTraining
        self.yTargetIndex = None
        self.dataLoaded = False
        self.targetsСolumns = []
        self.indicatorsСolumns= []
        self.X_TrainFiles = self.X_TestFiles = self.Y_TrainFiles = self.Y_TestFiles = []

        self.prepareXYtrainTestFiles()
        self.prepareTargetsIndicatorsColumnNames()

        self.pIndMeans = self.fIndMeans = np.zeros(len(self.indicatorsСolumns))
        self.pIndStds = self.fIndStds = np.ones(len(self.indicatorsСolumns))

        self.X1 = self.X2 = np.empty((0, len(self.indicatorsСolumns)), float)
        self.Y1 = self.Y2 = self.Y_True = np.empty((0, len(self.targetsСolumns)), float)

        if self.Configs.batchTraining:
            self.calcMeanStdForNormalization()

        if self.Configs.batchTraining:
            self.loadYTrueForBatchTraining()
        else:
            self.loadDataForNonBatchTrainingAndNormalize()

        ### load feature importance file
        self.fImportance = []
        self.fImportanceIndex = np.arange(0, len(self.indicatorsСolumns), 1)
        if os.path.exists(self.Configs.fImportanceFile):
            self.featuresImportance = pd.read_csv(self.Configs.fImportanceFile)
        else:
            print("Can not find file for features importance. Specified path: " + self.Configs.fImportanceFile)
            self.featuresImportance = pd.DataFrame()


    def loadYTrueForBatchTraining(self):
        self.Y_True = np.empty((0, len(self.targetsСolumns)), float)
        for yfile in self.Y_TestFiles:
            ff = np.load(yfile)
            self.Y_True = np.concatenate((self.Y_True, ff), axis=0)


    def prepareXYtrainTestFiles(self):
        print("Separate files on X,Y train/test")
        sd = datetime.strptime(self.Configs.startDay, '%Y-%m-%d')
        ed =  datetime.strptime(self.Configs.endDay, '%Y-%m-%d')

        dd = list(pd.date_range(sd, ed, freq='B'))
        dd = [str(x).split(" ")[0] for x in dd]

        if self.Configs.pathLabel != "":
            folders = os.listdir(self.Configs.defPath + "/" + self.Configs.instrument + "/" + self.Configs.pathLabel)
        else:
            folders = os.listdir(self.Configs.defPath + "/" + self.Configs.instrument)
        availableDates = [x for x in folders if len(re.findall(r"^\d+-\d+-\d+$", x)) > 0]

        dates = np.intersect1d(dd, availableDates)
        Xfiles, Yfiles = self.makeTrainTestFilesPaths(dates=dates, defPath=self.Configs.defPath, pathLabel=self.Configs.pathLabel, instrument=self.Configs.instrument)
        self.X_TrainFiles, self.X_TestFiles = self.splitTrainTestDates(interleaveSplit= self.Configs.interleaveSplit, dates=Xfiles, test2train=self.Configs.test2train)
        self.Y_TrainFiles, self.Y_TestFiles = self.splitTrainTestDates(interleaveSplit= self.Configs.interleaveSplit, dates=Yfiles, test2train=self.Configs.test2train)


    @staticmethod
    def splitTrainTestDates(interleaveSplit, dates, test2train):
        if interleaveSplit:
            train = []
            test = []
            for i, date in enumerate(dates):
                if (i+1) % int(1 / test2train) > 0:
                    train.append(date)
                else:
                    test.append(date)
        else:
            pTest = int(len(dates) * test2train)
            train = dates[:-pTest]
            test = dates[-pTest:]
        return train, test

    @staticmethod
    def makeTrainTestFilesPaths(dates, defPath, pathLabel, instrument):
        Xfiles = []
        Yfiles = []
        for date in dates:
            if pathLabel != "":
                xPath = defPath + "/" + instrument + "/" + pathLabel + "/" + date + "/indicators.npy"
                yPath = defPath + "/" + instrument + "/" + pathLabel + "/" + date + "/targets.npy"
            else:
                xPath = defPath + "/" + instrument + "/" + date + "/indicators.npy"
                yPath = defPath + "/" + instrument + "/" + date + "/targets.npy"
            if os.path.exists(xPath):
                Xfiles.append(xPath)
            if os.path.exists(yPath):
                Yfiles.append(yPath)
        return Xfiles, Yfiles


    def prepareTargetsIndicatorsColumnNames(self):
        print("Combine all files paths")

        if self.Configs.pathLabel != "":
            ftargetsСolumnsPath = self.Configs.defPath + "/" + self.Configs.instrument + "/" + self.Configs.pathLabel + "/targetsColumnNames.txt"
            findicatorsСolumnsPath = self.Configs.defPath + "/" + self.Configs.instrument + "/" + self.Configs.pathLabel + "/indicatorsColumnNames.txt"
        else:
            findicatorsСolumnsPath = self.Configs.defPath + "/" + self.Configs.instrument + "/indicatorsColumnNames.txt"
            ftargetsСolumnsPath = self.Configs.defPath + "/" + self.Configs.instrument + "/targetsColumnNames.txt"

        if os.path.exists(findicatorsСolumnsPath):
            findicatorsСolumns = open(findicatorsСolumnsPath, "r")
            self.indicatorsСolumns = findicatorsСolumns.readlines()
            self.indicatorsСolumns = [x.strip() for x in self.indicatorsСolumns]
        else:
            print("Couldn`t find path: " + findicatorsСolumnsPath)

        if os.path.exists(ftargetsСolumnsPath):
            ftargetsСolumns = open(ftargetsСolumnsPath, "r")
            self.targetsСolumns = ftargetsСolumns.readlines()
            self.targetsСolumns = [x.strip().replace(".", "") for x in self.targetsСolumns]

            self.yTargetIndex = self.targetsСolumns.index(self.Configs.target)
        else:
            print("Couldn`t find path: " + ftargetsСolumnsPath)


    def calcMeanStdForNormalization(self):
        st = datetime.now()
        indSumsp = np.zeros(len(self.indicatorsСolumns))
        indSums2 = np.zeros(len(self.indicatorsСolumns))
        indCounts = 0

        for xFile, yFile in zip(self.X_TrainFiles, self.Y_TrainFiles):
            xnfile = np.load(xFile)

            indSums = indSumsp + xnfile.sum(axis=0)
            indSums2 = indSums2 + (xnfile**2).sum(axis=0)
            indCounts = indCounts + xnfile.shape[0]

        self.IndMeans = indSums / indCounts
        self.IndStds = np.sqrt(1.0 / indCounts * (indSums2 - 1.0 / indCounts * indSums * indSums))

        print("Calculation means and stds on train set took: %s" % (datetime.now() - st))


    def saveTrainTestDates(self, modelsPath):
        np.savetxt(modelsPath + '/TrainFilesNames.txt', np.array(self.X_TrainFiles), fmt="%s")
        np.savetxt(modelsPath + '/TestFilesNames.txt', np.array(self.X_TestFiles), fmt="%s")


    def saveMeansStd(self, modelsPath):
        if self.Configs.saveMeansStd:
            np.savetxt(modelsPath + '/Means.txt', self.IndMeans)
            np.savetxt(modelsPath + '/Std.txt', self.IndStds)


    def loadDataForNonBatchTrainingAndNormalize(self):
        print("Load all data into memory, filter, and separate into variables")
        X1 = np.empty((0, len(self.indicatorsСolumns)), float)
        for xtrainfile in self.X_TrainFiles:
            xtrainArray = np.load(xtrainfile)
            X1 = np.append(X1, xtrainArray, axis=0)

        X2 = np.empty((0, len(self.indicatorsСolumns)), float)
        for xtestfile in self.X_TestFiles:
            xtestArray = np.load(xtestfile)
            X2 = np.append(X2, xtestArray, axis=0)

        Y1 = np.empty((0, len(self.targetsСolumns)), float)
        for ytrainfile in self.Y_TrainFiles:
            ytrainArray = np.load(ytrainfile)
            Y1 = np.append(Y1, ytrainArray, axis=0)

        Y2 = np.empty((0, len(self.targetsСolumns)), float)
        for ytestfile in self.Y_TestFiles:
            ytestArray = np.load(ytestfile)
            Y2 = np.append(Y2, ytestArray, axis=0)

        self.X1 = X1
        self.Y1 = Y1
        self.X2 = X2
        self.Y2 = Y2

        self.IndMeans = np.mean(self.X1, axis=0)
        self.IndStds = np.std(self.X1, axis=0)
        for i in range(self.X1.shape[1]):
            self.X1[:, i] = (self.X1[:, i] - self.IndMeans[i]) / self.IndStds[i]
            self.X2[:, i] = (self.X2[:, i] - self.IndMeans[i]) / self.IndStds[i]

        print("\nX1 shape: " + str(self.X1.shape))
        print("X2 shape: " + str(self.X2.shape))

        #del meansp, stdsp, meansf, stdsf
