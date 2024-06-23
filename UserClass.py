from sklearn.svm import OneClassSVM
from numpy import array

class UserModel():
    
    def __init__(self,Name,Password):
        self.AccountName = Name
        self.AccountPassword = Password
        self.TrainData    = []
        self.typing_speed = []
        self.typing_accuracy = []
        
    def AddTrainSet(self,TrainSet):
        self.TrainData = self.TrainData + TrainSet

    def ClearData(self,password):
        self.AccountPassword = password
        self.TrainData    = []
        self.typing_speed = []
        self.typing_accuracy = []

        
    def AddTypingMetrics(self, speed, accuracy):
        self.typing_speed.append(speed)
        self.typing_accuracy.append(accuracy)
        
    def CreateModel(self):
        hold = []
        sz = len(self.AccountPassword)*2

        for j in range(len(self.TrainData)):
            hold.append(array(self.TrainData)[j][sz:])
        return OneClassSVM(kernel = 'rbf',gamma="auto").fit(array(hold))
        
        
    
        
    
    
    