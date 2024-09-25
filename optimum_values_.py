import pandas as pd
import numpy as np

from caimcaim.caimcaim.caimcaim import CAIMD


class ExtractOptimumRange:
    def __init__(self, data, risk_map):

        self.data = data
        self.risk_map = risk_map

        self.columns  = { "metric1": "Fan Speed", "metric2": "Vibration Level", "metric3": "Refrigerant Pressure"
            , "metric4": "Humidity Level", "metric5": "Airflow Rate", "metric6": "Electrical Voltage"
            , "metric7": "Current Draw", "metric8": "Component Temperature", "metric9": "Filter Condition" }

        self.optimum_ranges = dict()

    @staticmethod
    def acquire_optimum_value(x):

        if isinstance(x, float):
            return x

        return (x.right - x.left)/2

    @staticmethod
    def get_failure_rate(x):
        return sum(x) / len(x)

    def set_optimum_ranges(self):

        test_data = self.data

        x = test_data.drop("failure", axis=1)
        y = test_data["failure"]

        x.drop(["device", "duration"], axis=1, inplace=True)
        # x.columns = list(self.columns.values())

        caim = CAIMD()

        caim.fit_transform(x, y)

        bins = caim.split_scheme

        index = 0
        for column in x.columns:

            if index in bins:
                new_dict = test_data.groupby(pd.cut(x[column],
                                                    bins=bins[index]))["failure"].apply(self.get_failure_rate).to_dict()

                self.optimum_ranges[column] = sorted(new_dict)[0]

            else:

                self.optimum_ranges[column] = 0

            index += 1


    def find_optimum_values(self):

        for col, range in self.optimum_ranges.items():
            self.optimum_ranges[col] = self.acquire_optimum_value(range)

    def return_optimum_values(self):
        pass