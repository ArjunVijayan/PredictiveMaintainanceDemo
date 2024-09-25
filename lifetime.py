import pandas as pd
import numpy as np

from sksurv.nonparametric import kaplan_meier_estimator

def AverageLifetime(df):
    x, y = kaplan_meier_estimator(df["failure"].astype("bool"), df["duration"])

    values = list(zip(x, y))
    columns = ["time", "prob_survival"]

    surv = pd.DataFrame(values, columns=columns)

    return  surv



