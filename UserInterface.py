import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from operator import mul
from sklearn.cluster import KMeans,Birch
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from PyQt5.QtWidgets import QMainWindow,QMessageBox,QApplication
from PyQt5.uic import loadUi
from time import time
from numpy import load,subtract,save,array2string,array
import numpy as np
from UserClass import UserModel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC,OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix,f1_score
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from scipy.stats import multivariate_normal
from json import dumps
from sys import argv



code = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()"
rdict = dict([ (x[1],x[0]) for x in enumerate(code) ])


class AccountWindow(QMainWindow):
    
    def __init__(self):
        
        QMainWindow.__init__(self)
        loadUi("UserInterface.ui",self)
        self.setWindowTitle("User authentication using Keystroke Dynamics")
        self.setFixedSize(400, 700) 
    
        self.ErrorMessage = QMessageBox()
        self.RegisterButton.clicked.connect(self.Register)
        self.LoginButton.clicked.connect(self.Login)
        self.ResetButton.clicked.connect(self.Reset)
        self.TrainButton.clicked.connect(self.Train)
        self.TestButton.clicked.connect(self.Predict)
        self.CompareAll.clicked.connect(self.Compare)
        self.ExportButton.clicked.connect(self.SessionExport)
        self.BackButton.clicked.connect(self.ToggleBack)
        self.ChangePasswordtext.clicked.connect(self.ChangePassword)

       
        self.ChangePasswordtext.hide()
        self.label_chp.hide()
        self.NewPasswordText.hide()
        self.isChangePasswordState = False 
        
        self.TimePressed  = []
        self.TimeReleased = []
        self.TrainSet     = []
        self.Dwell        = []
        self.Flight       = []
        self.ID = -1
        self.String = ""
        self.total_characters = ""
        self.typing_speed = 0
        self.typing_accuracy = 0
        self.PasswordText_3.setText(self.String)
        self.Reset()
        try:
            self.Accounts = load("Accounts\Accounts.npy",allow_pickle=True).tolist()
        except:
            self.Accounts = []
        
        
          
    
    def SessionExport(self):
        if self.ID < 0:
            self.ErrorMessage.setIcon(QMessageBox.Information)
            self.ErrorMessage.setText("Your are not logged in")
            self.ErrorMessage.setWindowTitle("Warning!")
            self.ErrorMessage.exec_()
        else:
            with open("Sessions.json", "w") as outfile: 
                outfile.write(dumps(self.Accounts[self.ID].TrainData)) 

        
    
    
    def ToggleBack(self):
        if not self.isChangePasswordState:
            # Switch to "Change Password" state
            self.BackButton.setText("Back")
            self.LoginButton.hide()
            self.ChangePasswordtext.show()
            self.NewPasswordText.show()
            self.label_chp.show()
        else:
            # Switch back to previous state
            self.BackButton.setText("Change Password")
            self.LoginButton.show()
            self.ChangePasswordtext.hide()
            self.NewPasswordText.hide()
            self.label_chp.hide()
            # Clear the new password textbox when switching back
            self.NewPasswordText.clear()
        self.isChangePasswordState = not self.isChangePasswordState
    
    def ChangePassword(self):
            if self.ID < 0:
                self.ErrorMessage.setIcon(QMessageBox.Information)
                self.ErrorMessage.setText("Your are not logged in")
                self.ErrorMessage.setWindowTitle("Warning!")
                self.ErrorMessage.exec_()
            if self.ID >= 0:
                if self.UserIDText_2.text() == self.Accounts[self.ID].AccountName and self.Accounts[self.ID].AccountPassword == self.PasswordText_2.text() and len(self.NewPasswordText.text()) >= 6:
                    self.Accounts[self.ID].ClearData(self.NewPasswordText.text())
                    
                    self.LoginText.setText("Password changed Successfully")
                else:
                    self.ErrorMessage.setIcon(QMessageBox.Information)
                    self.ErrorMessage.setText("Invalid credentials!")
                    self.ErrorMessage.setWindowTitle("Warning!")
                    self.ErrorMessage.exec_()    
                
            else:
                self.ErrorMessage.setIcon(QMessageBox.Information)
                self.ErrorMessage.setText("Your Password is Wrong")
                self.ErrorMessage.setWindowTitle("Warning!")
                self.ErrorMessage.exec_()




    
    def Compare(self):
        
        
        X = []
        y = []
        self.clf = SVC(kernel=self.comboBox.currentText())
        
        for i in range(len(self.Accounts)):
            if self.CompareText.text() == self.Accounts[i].AccountPassword:
                hold = []
                for k in range(len(self.Accounts[i].TrainData)):
                    hold.append(self.Accounts[i].TrainData[k][16:])
                X = X + hold
                for x in range(len(self.Accounts[i].TrainData)):
                    y.append(self.Accounts[i].AccountName)
        
        


        if X == []:
            self.ErrorMessage.setIcon(QMessageBox.Information)
            self.ErrorMessage.setText("There are no passwords")
            self.ErrorMessage.setWindowTitle("Warning!")
            self.ErrorMessage.exec_()  
        elif len(list(set(y))) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)       
            self.clf.fit(X_train,y_train)       
            arr = array2string(confusion_matrix(y_test,self.clf.predict(X_test)))
            self.CompareAllText.setText(arr +" "+str(round(self.clf.score(X_test,y_test),2)))
        else:
            self.CompareAllText.setText("There are not enough imposter data")
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def ProcessData(self):
        
        self.Dwell = subtract(self.TimeReleased,self.TimePressed).tolist()

        for i in range(len(self.TimePressed)-1):
                self.Flight.append(self.TimePressed[i+1] - self.TimeReleased[i])
            
        total_time = max(self.TimeReleased) - min(self.TimePressed)
        self.typing_speed = len(self.String) / total_time


        
        correct_characters = len(self.String)
        self.typing_accuracy = round(correct_characters /len(self.total_characters),2) if len(self.total_characters) > 0 else 0

        
        value = min(self.TimePressed) 
        for i in range(len(self.TimePressed)):
            self.TimePressed[i]  = self.TimePressed[i] - value
        value = min(self.TimeReleased) 
        for i in range(len(self.TimeReleased)):
            self.TimeReleased[i] =self.TimeReleased[i] - value
                
        self.TimePressed  = [ round(x,2) for x in self.TimePressed  ]
        self.TimeReleased = [ round(x,2) for x in self.TimeReleased ]
        self.Dwell        = [ round(x,2) for x in self.Dwell        ]
        self.Flight       = [ round(x,2) for x in self.Flight       ]
        # self.typing_speed        = [ round(x,2) for x in self.typing_speed        ]
        # self.typing_accuracy       = [ round(x,2) for x in self.typing_accuracy       ]
    
    def Predict(self):


        if self.ID < 0:
            self.ErrorMessage.setIcon(QMessageBox.Information)
            self.ErrorMessage.setText("You are not logged in")
            self.ErrorMessage.setWindowTitle("Warning!")
            self.ErrorMessage.exec_()
        elif  self.String == self.Accounts[self.ID].AccountPassword:
            y = []
            for i in range(len(self.Accounts)):
                if self.Accounts[self.ID].AccountPassword == self.Accounts[i].AccountPassword:
                    for x in range(len(self.Accounts[i].TrainData)):
                        y.append(self.Accounts[i].AccountName)


            sts = len(list(set(y)))

            self.ProcessData()
            

            
                

             

            Xset = []
            Yset = []
            sz = len(self.Accounts[self.ID].AccountPassword)*2

            for j in range(len(self.Accounts[self.ID].TrainData)):
                Xset.append(array(self.Accounts[self.ID].TrainData)[j][sz:])
                Yset.append(1)
            
            Xset = array(Xset)
            Yset = array(Yset)

            

            trainx, testx, trainy, testy = train_test_split(Xset, Yset, test_size=0.3, random_state=2)

            trainx = array(trainx)

            X = []
            multiy = []
            multi2y = []

            if sts > 1:
                
                for i in range(len(self.Accounts)):
                    if self.Accounts[self.ID].AccountPassword == self.Accounts[i].AccountPassword and self.ID != i:
                        hold = []
                        for k in range(len(self.Accounts[i].TrainData)):
                            hold.append(self.Accounts[i].TrainData[k][16:])
                        X = X + hold
                        for x in range(len(self.Accounts[i].TrainData)):
                            multiy.append(-1)
                            multi2y.append(0)
                X = array(X)
                multiy = array(multiy)
                multi2y = array(multi2y)    


                testx = np.concatenate((testx,X))
                testymone = np.concatenate((testy,multiy))
                testymzero = np.concatenate((testy,multi2y))

            if sts == 1:
                testymone  = testy
                testymzero = testy

            Osvm = OneClassSVM(kernel = 'rbf',gamma="auto").fit(trainx)
            Ypredict = Osvm.predict(testx)
            score = f1_score(testymone, Ypredict, pos_label=1)


            kmeans = KMeans(n_clusters=2, random_state=0).fit(trainx)
            Ypredict = kmeans.predict(testx)
            score1 = f1_score(testymzero, Ypredict, pos_label=1)
                

            brc = Birch(n_clusters=2,threshold=0.01).fit(trainx)
            Ypredict = brc.predict(testx)
            sil2 = str(silhouette_score(trainx, brc.labels_))
            score2 = f1_score(testymzero, Ypredict, pos_label=1)

            IsF = IsolationForest(contamination=0.01)
            IsF.fit(trainx)
            Ypredict = IsF.predict(testx)
            score3 = f1_score(testymone, Ypredict, pos_label=1)
                

            ev = EllipticEnvelope(contamination=0.01)
            ev.fit(trainx)
            Ypredict = ev.predict(testx)
            score4 = f1_score(testymone, Ypredict, pos_label=1)

            results_array = []
            results_array.append(1 if Osvm.predict([self.Dwell+self.Flight]) == 1 else 0)
            results_array.append(1 if kmeans.predict([self.Dwell+self.Flight]) == 1 else 0)
            results_array.append(1 if brc.predict([self.Dwell+self.Flight]) == 1 else 0)
            results_array.append(1 if IsF.predict([self.Dwell+self.Flight]) == 1 else 0)
            results_array.append(1 if ev.predict([self.Dwell+self.Flight]) == 1 else 0)
            pred_res = sum(1 for result in results_array if result == 1)

            if Osvm.predict([self.Dwell+self.Flight]) == 1:
                OsvmResult = 'pass'
               
            else:
                OsvmResult = 'fail'

            if kmeans.predict([self.Dwell+self.Flight]) == 1:
                kmResult = 'pass'
             
            else:
                kmResult = 'fail'

            if brc.predict([self.Dwell+self.Flight]) == 1:
                brcResult = 'pass'
               
            else:
                brcResult = 'fail'

            if IsF.predict([self.Dwell+self.Flight]) == 1:
                IsFResult = 'pass'
               
            else:
                IsFResult = 'fail'

            if ev.predict([self.Dwell+self.Flight]) == 1:
                evResult = 'pass'
                
            else:
                evResult = 'fail'

            # print(count_ones)

            mean_f1 = round((score+score1+score2+score3+score4)/5,2)
            # norm_f1 = round((mean_f1/max(score,score1,score2,score3,score4)),2)
            norm_acc = round(self.typing_accuracy/max(self.Accounts[self.ID].typing_accuracy),2)
            if round(self.typing_speed/max(self.Accounts[self.ID].typing_speed),2) <= 1.00:
                norm_speed = round(self.typing_speed/max(self.Accounts[self.ID].typing_speed),2)
            elif 1.00 < round(self.typing_speed/max(self.Accounts[self.ID].typing_speed),2) <= 1.50:
                norm_speed = 0.5
            else:
                norm_speed = 0

            model_pred = round(pred_res/5,2)
            # print(model_pred)
            normacc2 = round((sum(self.Accounts[self.ID].typing_accuracy)//len(self.Accounts[self.ID].typing_accuracy))/max(self.Accounts[self.ID].typing_accuracy),2)
            normspeed2 = round((sum(self.Accounts[self.ID].typing_speed)//len(self.Accounts[self.ID].typing_speed))/max(self.Accounts[self.ID].typing_speed),2)

            if pred_res > 2:
                model_decision = "Accepted"
            elif pred_res == 2 and round((norm_acc + norm_speed + model_pred)/3,2) > 0.7:
                model_decision = "Partially Accepted"
            else:
                model_decision = "Rejected"

            self.TrainText.setText("Score/Model"+" \n" + str(round(score,2)) + " Osvm: "+ OsvmResult + " \n"+str(round(score1,2)) +" Km: " + kmResult + " \n"+str(round(score2,2)) +" Brc: "+ brcResult + " \n " +str(round(score3,2)) + " ISF: "+ IsFResult + " \n"+str(round(score4,2)) +" Ev: "+ evResult + f' \n Combined Score = {round((norm_acc + norm_speed + model_pred)/3,2)}\n' + f"{model_decision}")

          

            

            self.Reset()


        else:
            self.ErrorMessage.setIcon(QMessageBox.Information)
            self.ErrorMessage.setText("Your password is wrong")
            self.ErrorMessage.setWindowTitle("Warning!")
            self.ErrorMessage.exec_()  
        
        
        
        
        
        
        
    def Train(self):
        
        if self.ID < 0:
            self.ErrorMessage.setIcon(QMessageBox.Information)
            self.ErrorMessage.setText("Your are not logged in")
            self.ErrorMessage.setWindowTitle("Warning!")
            self.ErrorMessage.exec_()
        elif  self.String == self.Accounts[self.ID].AccountPassword:
            
            
            
            self.ProcessData()
                
            
            self.Accounts[self.ID].AddTrainSet([self.TimePressed+self.TimeReleased+self.Dwell+self.Flight])
            self.Accounts[self.ID].AddTypingMetrics(self.typing_speed, self.typing_accuracy)
            save("Accounts\Accounts", self.Accounts,allow_pickle=True)
            
            
            
            
            size = len(self.Accounts[self.ID].TrainData)
            self.TrainText.setText("Your data base increased to {} \n".format(size) + f"Avg speed: {round(sum(self.Accounts[self.ID].typing_speed)/(len(self.Accounts[self.ID].typing_speed)),2)}")
            self.Reset()
        else:
            self.ErrorMessage.setIcon(QMessageBox.Information)
            self.ErrorMessage.setText("Your password is wrong")
            self.ErrorMessage.setWindowTitle("Warning!")
            self.ErrorMessage.exec_()  
        
        
        
        
        
        
        
    
    def Reset(self):
        self.String = ""
        self.total_characters = ""
        self.TimePressed = []
        self.TimeReleased = []
        self.Dwell        = []
        self.Flight       = []
        self.PasswordText_3.setText(self.String)
    
    
    def Register(self):
        
        if len(self.UserIDText_1.text()) >= 6 and len(self.PasswordText_1.text()) >= 6:
            check = False
            for i in range(len(self.Accounts)):
                if self.UserIDText_1.text() == self.Accounts[i].AccountName:
                    check = True
                
            if check == True:
                self.ErrorMessage.setIcon(QMessageBox.Information)
                self.ErrorMessage.setText("There is a person has that account name")
                self.ErrorMessage.setWindowTitle("Warning!")
                self.ErrorMessage.exec_()       
            else:
                person = UserModel(self.UserIDText_1.text(),self.PasswordText_1.text())
                self.Accounts.append(person)
                save("Accounts\Accounts", self.Accounts,allow_pickle=True)
                self.RegistrationText.setText("Registration is completed. Thank you " + person.AccountName) 
        
        else:
            self.ErrorMessage.setIcon(QMessageBox.Information)
            self.ErrorMessage.setText("Your ID and Password Must be greater than 6 letters")
            self.ErrorMessage.setWindowTitle("Warning!")
            self.ErrorMessage.exec_()    
        
    
    def Login(self):
        ID = -1
        for i in range(len(self.Accounts)):
            if self.UserIDText_2.text() == self.Accounts[i].AccountName:
                ID = i
        if ID >= 0:
            if self.Accounts[ID].AccountPassword == self.PasswordText_2.text():
                self.ID = ID
                self.LoginText.setText("You are logged in " + self.Accounts[ID].AccountName + ". Welcome " )
                
            else:
                self.ErrorMessage.setIcon(QMessageBox.Information)
                self.ErrorMessage.setText("Your Password is Wrong")
                self.ErrorMessage.setWindowTitle("Warning!")
                self.ErrorMessage.exec_()
        else:
            self.ErrorMessage.setIcon(QMessageBox.Information)
            self.ErrorMessage.setText("There is no account has this name")
            self.ErrorMessage.setWindowTitle("Warning!")
            self.ErrorMessage.exec_()
    
    def keyPressEvent(self,event):
        tm = time()
        
        try:
            convertion = rdict[event.text()]
            self.String += event.text()
            self.total_characters += event.text()
            self.TimePressed.append(tm)
        except:
            if 16777219 == event.key():
                if len(self.String) > 0:
                    self.TimePressed.pop()
                    self.String = self.String[:-1]
                else:
                    self.Reset()
            if 16777220 == event.key():
                self.Train()
        self.PasswordText_3.setText(self.String)
    
    
    def keyReleaseEvent(self,event):
        tm = time()
        try:
            convertion = rdict[event.text()]
            self.TimeReleased.append(tm)
        except:
            if 16777219 == event.key():
                if len(self.String) > 0:
                    self.TimeReleased.pop()
                else:
                    self.Reset()





app = QApplication(argv)
window = AccountWindow()
window.show()
app.exec_()
