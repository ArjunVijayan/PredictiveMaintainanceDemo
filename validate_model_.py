import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score

class ValidateModel:
    def __init__(self, model, time_window=30):
        self.train_data = None
        self.validation_data = None
        self.model = model
        self.time_window = time_window

    def read_csv(self):
        path = "data/"

        if self.time_window == 30:

            df_train = pd.read_csv(f"{path}df_train.csv")
            actual_results = pd.read_csv(f"{path}actual_results.csv")
            df_train.drop(["date", "failure"], axis=1, inplace=True)
        
        if self.time_window == 15:
            df_train = pd.read_csv(f"{path}df_train15.csv")
            actual_results = pd.read_csv(f"{path}actual_results15.csv")
            df_train.drop(["date", "failure"], axis=1, inplace=True)

        else:
            df_train = pd.read_csv(f"{path}df_train6.csv")
            actual_results = pd.read_csv(f"{path}actual_results6.csv")
            df_train.drop(["date", "failure"], axis=1, inplace=True)

        self.train_data = df_train
        self.actual_results = actual_results

    def compare_functions(self, survival_df):

        valid_df = self.actual_results.merge(survival_df, on="device", how="left")
        valid_df["prediction"] = valid_df["Expected Time"] >= valid_df["act_dur"]
        valid_df["prediction"] = valid_df["prediction"] & valid_df["failure"]

        return valid_df , confusion_matrix(valid_df["failure"], valid_df["prediction"])
    
    def train_and_validate(self):
        self.read_csv()
        survival_df = self.model.estimate_ttmf_(self.train_data)
        valid_df, cmatrix = self.compare_functions(survival_df)

        return valid_df, cmatrix, survival_df
    

