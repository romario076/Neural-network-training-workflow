

Check Configs in trainDNN.py
If you want to run multiple targets, check targets list in trainDNN.py before 'for' loop.

If you want to add your own new loss function, you can add it into lossFunctions.py, and then specify it in trainDNN.py Config class.

Run: python trainDNN.py

Run TensorBoard(in directory where trainDNN.py): tensorboard --logdir='TensorBoardLogs/' --host=10.12.1.59 --port=8999  (check if port not busy)