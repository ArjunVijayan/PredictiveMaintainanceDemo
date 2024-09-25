import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class AnomalyDetection:

    def __init__(self, data):
        self.data = data

        self.noc = None
        self.feature_names = None
        self.scaler = None

        self.model =None

        self.warning_thresh = 19.03
        self.failure_thresh = 20.00

    def identifyNOC(self):
        noc = self.data[self.data["failure"]==0].reset_index(drop=True)
        dat_ = noc.drop(["device", "failure", "duration"], axis=1)
        target_ = noc['failure']

        self.noc = dat_
        self.feature_names = dat_.columns

        return dat_
    
    def estimateKDE(self):

        dat_ = self.identifyNOC()
        scaler = MinMaxScaler()
        dat_ = scaler.fit_transform(dat_)
        model = KernelDensity()
        model.fit(dat_)

        scores = model.score_samples(dat_)

        self.scaler = scaler
        self.model = model
        return dat_, scores

    def predict_for_(self, record):
        x = self.scaler.transform(record[self.feature_names])
        score = self.model.score(x)

        if abs(score) > self.failure_thresh:
            return "Send Failure Alert"

        if abs(score) > self.warning_thresh:
            return "Send Warning"

        return "Normal Operation"
