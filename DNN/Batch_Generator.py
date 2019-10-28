
import keras
import numpy as np

class Batch_Generator(keras.utils.Sequence):
    def __init__(self, xFilenames, yFilenames, batch_size, yTargetIndex, batchNormalization, volumeLeftCol, means, stds):
        self.xFilenames = xFilenames
        self.yFilenames = yFilenames
        self.batch_size = batch_size
        self.yTargetIndex = yTargetIndex
        self.batchNormalization = batchNormalization
        self.means = means
        self.stds = stds
        self.volumeLeftCol = volumeLeftCol

    def __len__(self):
        return (np.ceil(len(self.xFilenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.xFilenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.yFilenames[idx * self.batch_size: (idx + 1) * self.batch_size]

        Xs = np.load(batch_x[0])
        for file_name in batch_x[1:]:
            Xs = np.append(Xs, np.load(file_name), axis=0)

        Ys = np.load(batch_y[0])
        for file_name in batch_y[1:]:
            Ys =  np.append(Ys, np.load(file_name), axis=0)

        Ys = Ys[:, self.yTargetIndex]
        Xs = Xs

        if not self.batchNormalization:
            Xs = (Xs - self.means) / self.stds

        return Xs, Ys
