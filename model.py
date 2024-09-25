import shap

import pandas as pd
import numpy as np

from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.util import Surv
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest


class FailureTimeModel:

    def __init__(self, data, id_column, duration_column, event_column):

        self.data = data

        self.id_column = id_column
        self.duration_column = duration_column
        self.event_column = event_column

        self.reference_point = 60

        self.model = None
        self.feature_names = None

    def set_shap_explanations(self, x_train, feature_names):

        explainer = shap.Explainer(self.model.predict, x_train, feature_names=feature_names)
        self.explainer = explainer

        return { "success": True }

    def extract_explanations(self, record):
        record = record[self.feature_names]
        return self.explainer(record), self.feature_names

    def extract_survival_time_(self, records):

        T = []
        P = []

        records_ = records.drop("device", axis=1)

        for i, row in records_.iloc[:, 1:].iterrows():
            diff = np.abs(row - 0.5)
            min_diff_idx = diff.idxmin()
            T.append(min_diff_idx)
            P.append(row[min_diff_idx])

        new_df = pd.DataFrame({
            'device': list(records["device"]),
            'Expected Time': T,
            'Score': P
        })

        return new_df

    def encode_data_(self, data=None, prediction=False):

        if data is not None:
            data = data

        else:
            data = self.data

        columns_ = set(data.columns)
        
        columns_to_encode_ = columns_ - set([self.event_column, self.duration_column])
        columns_to_encode_ = list(columns_to_encode_ - set(self.id_column))

        data_to_encode_ = data[columns_to_encode_]

        X_num = data_to_encode_.select_dtypes(exclude='object')
        X_cat = data_to_encode_.select_dtypes(include='object')

        if prediction:

            X_cat = pd.DataFrame(self.encoder.transform(X_cat)
            , columns=self.categorical_column_names)

            encoded_data = X_num.join(X_cat)
            encoded_data.fillna(0, inplace=True)

            for col_name in self.id_columns:
                encoded_data[col_name] = data[col_name]

            return encoded_data

        self.colums_to_encode = X_cat.columns

        enc = OneHotEncoder()
        X_cat = pd.DataFrame(enc.fit_transform(X_cat))

        self.encoder = enc

        categorical_columns = self.colums_to_encode

        X_cat = pd.DataFrame(X_cat, columns=enc.get_feature_names_out(categorical_columns))

        self.categorical_column_names = enc.get_feature_names_out(categorical_columns)

        encoded_data = X_num.join(X_cat)
        encoded_data.fillna(0, inplace=True)

        for col_name in self.id_columns:
            encoded_data[col_name] = data[col_name]

        return encoded_data


    def preprocess_data_(self, x, y=None):
        
        x_encoded = x
        x_processed = x_encoded.fillna(0)

        y_processed = None

        if y is not None:
            y_processed = Surv.from_dataframe(event=self.event_column
            , time=self.duration_column, data=y)
        
        return x_processed, y_processed

    def train_model_(self):

        id_columns = [self.id_column]
        target_columns = [self.event_column, self.duration_column]

        self.data = self.data.set_index(id_columns)
        self.unique_durations = list(self.data[self.duration_column].unique())

        x, y = self.data.drop(target_columns, axis=1), self.data[target_columns]
        x, y = self.preprocess_data_(x, y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        rsf = RandomSurvivalForest(n_estimators=300, min_samples_split=10, min_samples_leaf=15
        , n_jobs=-1, random_state=0)

        rsf.fit(x_train, y_train)

        concordance_score = rsf.score(x_test, y_test)

        feature_names = list(x_train.columns)

        self.feature_names = feature_names

        self.model = rsf

        reasoning_response = self.set_shap_explanations(x_train, feature_names)

        return rsf, concordance_score, reasoning_response

    def estimate_ttmf_(self, records):

        surv_func = self.estimate_survival_function_(records)

        survival_df = self.extract_survival_time_(surv_func)

        return survival_df

    def estimate_risk_df_(self, records):

        ids = records[self.id_column].values

        records_ = records.drop([self.duration_column, self.id_column, 'failure'], axis=1)

        x, _ = self.preprocess_data_(records_)
        surv = self.model.predict_survival_function(x, return_array=True)

        surv_func = pd.DataFrame(surv)
        surv_func = 1 - surv_func
 
        surv_func.columns = self.model.unique_times_
        risk_point =  surv_func.columns[self.reference_point]

        def return_risk_label(x):

            if x <= 0.50:
                return "LOW RISK"

            elif (x > 0.50) & (x <= 0.75):
                return "MEDIUM RISK"

            else:
                return "HIGH RISK"

        risk_df = pd.DataFrame()
        risk_df["device"] = pd.Series(ids)
        risk_df["score"] = surv_func[risk_point].values
        risk_df["risk"] = risk_df["score"].apply(return_risk_label)

        return risk_df

    def estimate_survival_function_(self, records):

        ids = records[self.id_column].values

        records_ = records.drop([self.duration_column, self.id_column], axis=1)

        x, _ = self.preprocess_data_(records_)
        surv = self.model.predict_survival_function(x, return_array=True)

        surv_func = pd.DataFrame(surv)
        surv_func = 1 - surv_func
 
        surv_func.columns = self.model.unique_times_
        columns = ["device"] + list(surv_func.columns)
        surv_func["device"] = pd.Series(ids)

        surv_func = surv_func[columns]

        return surv_func

    def rank_machine_failures_(self, records):

        ids = records[self.id_column].values

        records_ = records.drop([self.duration_column, self.id_column], axis=1)

        x, _ = self.preprocess_data_(records_)

        rank = self.model.predict(x)

        ranked_failures = pd.DataFrame()

        ranked_failures["ID"] = pd.Series(ids)

        ranked_failures["RiskScore"] = pd.Series(rank)

        ranked_failures.sort_values(by ="RiskScore", ascending=False, inplace=True)

        ranked_failures.reset_index(drop=True)

        return ranked_failures