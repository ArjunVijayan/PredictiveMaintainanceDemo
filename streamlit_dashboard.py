import copy
import shap
import random
import datetime

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st

from model import FailureTimeModel
from optimum_values_ import ExtractOptimumRange
from validate_model_ import ValidateModel
from sklearn.preprocessing import MinMaxScaler
from anomaly_detection import AnomalyDetection
from lifetime import AverageLifetime

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


sns.set_style("darkgrid")
plt.style.use("tableau-colorblind10")

def read_data():

    path = "data/"

    df = pd.read_csv(f"{path}processed_df.csv")

    observation_begining_date = datetime.date(year=2015, month=1, day=1)

    df_ = copy.deepcopy(df)
    df_['duration'] =  df_['duration'].apply(lambda x: datetime.timedelta(days=x))
    df_['duration'] = df_['duration'].apply(lambda x: observation_begining_date + x)

    return df_

def check_for_anom(x):

    val  = x["actual reading"]

    if val < x["Optimal Range - Lower Limit"]:
        return "LOW"

    elif val > x["Optimal Range - Upper Limit"]:
        return "HIGH"
    
    return "OPTIMAL"

def plot_bar(data, neg=False):

    if neg:
        col_code = '#009dd1'
    else:
        col_code = '#9e1b32'

    fig, ax = plt.subplots()
    data.sort_values(by="value", ascending=False)
    sns.barplot(data=data, x="value", y="variable", width=0.5, color=col_code)
    ax.bar_label(ax.containers[0])
    plt.xlabel("Feature Contribution")
    plt.ylabel("Metrics")
    return fig


def plot_bar_1(data):

    optimum_values = st.session_state.optimum_val
    actual_data = dict(zip(data["variable"].values, data["value"].values))

    res = pd.DataFrame()
    res["actual reading"] = pd.Series(actual_data)
    res["optimal value"] = pd.Series(optimum_values)

    st.session_state.optimal_range_table = res

    tile = st.container(border=True)
    tile.markdown("<h4 style='text-align: center; '>Sensor Readings</h4>", unsafe_allow_html=True)
    col1, col2 = tile.columns(2)

    with col1:
        tile = col1.container(border=True)
        tile.metric(label ="VOLTAGE" ,value = round(actual_data["volt"], 2))

        tile = col1.container(border=True)
        tile.metric(label ="RPM" ,value = round(actual_data["rotate"], 2))

    with col2:
        tile = col2.container(border=True)
        tile.metric(label ="PRESSURE" ,value = round(actual_data["pressure"], 2))

        tile = col2.container(border=True)
        tile.metric(label ="VIBRATION" ,value = round(actual_data["vibration"], 2))

def configure_model():

    tile = st.container(border=True)

    tile.header(":clipboard: Sample Data")

    path = "data/"

    df = pd.read_csv(f"{path}PdM_telemetry.csv")
    df_ = pd.read_csv(f"{path}processed_df.csv")

    df1 = pd.read_csv(f"{path}PdM_machines.csv")
    df2 = pd.read_csv(f"{path}PdM_failures.csv")

    tile.markdown("#### Telemetry Time Series Data")
    tile.dataframe(df, height=250, width=900)

    col1, col2 = tile.columns(2)

    with col1:
        st.markdown("#### Metadata of Machines")
        st.dataframe(df1, height=250, width=900)

    with col2:
        st.markdown("#### Machine Maintenance/Failures")
        st.dataframe(df2, height=250, width=900)

    df_ = copy.deepcopy(df_)
    df_.drop("date", axis=1, inplace=True)

    df_disp = df_

    st.session_state.timelag_data = df_disp
    tile.info("##### Data Description \n- **Telemetry Time Series Data**\t: It consists of hourly average of voltage, rotation, pressure, vibration collected from 100 machines for the year 2015.\n- **Metadata of Machines**\t: Model type & age of the Machines.\n- **Machine Maintenances / Failures**\t:  Each record represents replacement of a component due to failure or regular schedule.")
    tile = st.container(border=True)
    tile.text("Required Details for Model Training\n- 'device'\t : Unique identifier for each device.\n- 'duration'\t : Number of days from installation of the machine till its failure.\n- 'failure'\t : Binary indicator of failure (1 for failure and 0 otherwise).")

    id_column = "device"
    duration_column = "duration"
    event_column = "failure"

    train = st.button("Train Hazard Model")
    model = FailureTimeModel(df_disp, id_column=id_column
    , duration_column=duration_column, event_column=event_column)

    if train:
        with st.spinner("Please Wait.."):

            mod, score, reasoning_response = model.train_model_()

            st.session_state.cindex = round(score, 2)

            st.markdown("<h2 style='text-align: left; '>Training Response</h2>", unsafe_allow_html=True)

            st.json({"Training Status": "Success", "Reasoning Status": reasoning_response["success"]
            , "Concordance Score":round(score, 2)})

            st.info("In survival analysis, the concordance score (C-index) measures how well a model ranks survival times based on predicted risks. A score of 1 indicates perfect prediction, 0.5 suggests random prediction, and below 0.5 indicates poor prediction.")


    st.session_state.model = model
    st.session_state.data = df_disp
    st.session_state.id_column = id_column
    st.session_state.duration_col = duration_column
    st.session_state.event_column = event_column

def rank_risk(x):
    if x <= 0.30:
        return "LOW RISK"

    elif (x > 0.30) & (x <= 0.60):
        return "MEDIUM RISK"

    else:
        return "HIGH RISK"

def expected_failure_times():

    model = st.session_state.model
    data = st.session_state.data

    event_column = st.session_state.event_column
    id_column = st.session_state.id_column
    duration_column = st.session_state.duration_col

    data_ = data[data['failure']==1].reset_index(drop=True)
    data_ = data_.groupby(id_column).first().reset_index()

    risk_df = model.estimate_risk_df_(data_)
    dat_ = data_.drop(event_column, axis=1)

    individuals = list(dat_[id_column].unique())
    individuals_selected = individuals[0:6]

    st.session_state.individuals_selected = individuals_selected

    n = len(individuals_selected)

    col1, col2, col3 = st.columns([1, 8, 1])

    with col2:

        if n >= 1:
            dat_ = dat_[dat_[id_column].isin(individuals_selected)]
            risk_df = risk_df[risk_df[id_column].isin(individuals_selected)]
            st.markdown("#### Expected Failure Time")    
            
            st.markdown("Selected Individuals")
            new_columns = { "metric1": "Fan Speed", "metric2": "Vibration Level", "metric3": "Refrigerant Pressure"
                , "metric4": "Humidity Level", "metric5": "Airflow Rate", "metric6": "Electrical Voltage"
                , "metric7": "Current Draw", "metric8": "Component Temperature", "metric9": "Filter Condition" }

            data_disp = dat_.rename(columns=new_columns)
            st.dataframe(data_disp.drop(duration_column, axis=1))

    st.markdown("#### Estimate the expected device failure time and assess the immediate risk levels of individual devices.")
  
        
    ranked_df = model.rank_machine_failures_(dat_)        
    survival_df = model.estimate_ttmf_(dat_)

    ranked_df["Expeted_FT"] = survival_df["Expected Time"]

    risk_map = risk_df.groupby("device")["risk"].last().to_dict()
    ranked_df["Immediate_Risk_Level"] = ranked_df["ID"].map(risk_map)

    model = st.session_state.model
    dat_ = dat_.groupby("device").last().reset_index()

    st.session_state.risk_score = ranked_df.groupby("ID")["RiskScore"].last().to_dict()
    st.session_state.result_df = ranked_df

def visualize_results():
    
    model = st.session_state.model
    data = st.session_state.data

    event_column = st.session_state.event_column
    id_column = st.session_state.id_column
    duration_column = st.session_state.duration_col

    data = data.groupby(id_column).first().reset_index()
    dat_ = data.drop(event_column, axis=1)

    individuals = list(dat_[id_column].unique())
    individuals_selected = individuals[:5] + individuals[-5:]

    n = len(individuals_selected)

    tile = st.container(border=True)

    if n >= 1:
        dat_ = dat_[dat_[id_column].isin(individuals_selected)]
        
        survival_func = model.estimate_survival_function_(dat_)

        ids = survival_func[id_column]

        survival_func = survival_func.drop(id_column, axis=1)
        survival_func_t = survival_func.transpose()
        
        survival_func_t.columns = ids

        tile.markdown("<h4 style='text-align: left; '>Hazard Function</h4>", unsafe_allow_html=True)

        tile.write("Visualize how the probability of device failure evolves over time for each device by plotting the hazard function, which represents the instantaneous failure rate at any given time.")

        tile.line_chart(survival_func_t)

        tile.info("The hazard function plot depicts the time until failure on the x-axis and the probability of failure occurring at that specific point in time on the y-axis.")

def forge_dashboard():

    model = st.session_state.model
    data = st.session_state.data

    data = data.groupby("device").last().reset_index()

    risk_df = model.estimate_risk_df_(data)

    all_devices = risk_df["device"].unique().tolist()

    total_devices = risk_df.shape[0]
    lr_devices = risk_df[risk_df["risk"]=="LOW RISK"].shape[0]
    lr_unique = risk_df[risk_df["risk"]=="LOW RISK"]["device"].unique().tolist()

    mr_devices = risk_df[risk_df["risk"]=="MEDIUM RISK"].shape[0]
    mr_unique = risk_df[risk_df["risk"]=="MEDIUM RISK"]["device"].unique().tolist()

    hr_devices = risk_df[risk_df["risk"]=="HIGH RISK"].shape[0]
    hr_unique = risk_df[risk_df["risk"]=="HIGH RISK"]["device"].unique().tolist()

    tile = st.container(border=True)
    data = pd.DataFrame([["0-LOW RISK", lr_devices],["1-MEDIUM RISK", mr_devices], ["2-HIGH RISK", hr_devices]], columns=["risk", "count"])

    tile.markdown("<h4 style='text-align: left; '>Risk Allocation</h4>", unsafe_allow_html=True)

    tile.markdown("Segment devices into 'LOW,' 'MEDIUM,' and 'HIGH' risk categories to prioritize maintenance and optimize resource allocation.")
    tile.markdown("\n\n\n")
    tile.area_chart(data = data, x="risk", y="count", height=300)

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        tile = col1.container(border=True)
        tile.metric(label ="ALL DEVICES" ,value = total_devices, help = "Total number of devices.")
        tile.selectbox("", all_devices)

    with col2:
        tile = col2.container(border=True)
        tile.metric(label ="LOW RISK" ,value = round(lr_devices), help = "Devices with less than 50% chance of failure within 60 hours.")
        tile.selectbox("", lr_unique)

    with col3:
        tile = col3.container(border=True)
        tile.metric(label ="MEDIUM RISK" ,value = round(mr_devices), help = "Devices with 50-80% chance of failiure within 60 hours.")
        tile.selectbox("", mr_unique)

    with col4:
        tile = col4.container(border=True)
        tile.metric(label ="HIGH RISK" ,value = round(hr_devices), help = "Devices with more than 80% chance of failure within 60 hours.")
        tile.selectbox("", hr_unique)

def provide_reasonings():

    model = st.session_state.model
    
    indiv = st.session_state.individuals_selected
    data = st.session_state.data
    id_column = st.session_state.id_column

    res_df = st.session_state.result_df

    records = data[data[id_column].isin(indiv)]
    records = records.groupby(id_column).last().reset_index()

    opt = ["None"] + list(records[id_column].unique())
    sel_indiv = st.selectbox("Select a device", opt)

    flag = st.button("Provide Reasons")

    if flag:

        record = records[records[id_column]==sel_indiv]
        res = res_df[res_df["ID"]==sel_indiv]

        with st.spinner("Please wait..."):

            st.markdown(f"<h1 style='text-align: center; '>Device - {sel_indiv}</h1>", unsafe_allow_html=True)

            explanations, _ = model.extract_explanations(record)

            new_columns = { "metric1": "Fan Speed", "metric2": "Vibration Level", "metric3": "Refrigerant Pressure"
            , "metric4": "Humidity Level", "metric5": "Airflow Rate", "metric6": "Electrical Voltage"
            , "metric7": "Current Draw", "metric8": "Component Temperature", "metric9": "Filter Condition" }

            record_ = record.drop(["failure", "duration", "device"], axis=1)

            tile = st.container(border=True)

            plot_bar_1(record_.melt())

            exp_df = pd.DataFrame(explanations.values)
            exp_df.columns = record_.columns
            exp_df["device"] = pd.Series(indiv)

            exp_df = exp_df.set_index("device").melt()

            tile = st.container(border=True)
            tile.markdown("<h4 style='text-align: center; '>Model Predictions</h4>", unsafe_allow_html=True)

            col1, col2 = tile.columns(2)

            with col1:
                tile = col1.container(border=True)
                tile.metric(label ="Predicted Failure Time" ,value = res["Expeted_FT"])

            with col2:
                tile = col2.container(border=True)
                tile.metric(label ="Predicted Risk Level" ,value = res["Immediate_Risk_Level"].values[0])

            exp_df["value"] = exp_df["value"].apply(lambda x: round(x, 2))

            tile1 = st.container(border=True)
            tile1.markdown("<h4 style='text-align: center; '>Analyze how each metric affects risk prediction</h4>", unsafe_allow_html=True)

            positive_features = exp_df[exp_df["value"] > 0]
            negative_features = exp_df[exp_df["value"] < 0]

            pos_sum = positive_features["value"].sum()
            neg_sum = negative_features["value"].sum()

            positive_features["value"] = positive_features["value"].apply(lambda x: round(x/pos_sum, 2))
            negative_features["value"] = negative_features["value"].apply(lambda x: round(x/neg_sum, 2))

            positive_features.sort_values(by="value", ascending=False, inplace=True)
            negative_features.sort_values(by="value", ascending=False, inplace=True)

            col1, col2 = tile1.columns(2)

            with col2:
                tile = st.container(height=300, border=True)
                tile.markdown("- Risk Inducing Factors")
                fig = plot_bar(positive_features)
                tile.pyplot(fig)

            with col1:
                tile = st.container(height=300, border=True)
                tile.markdown("- Risk Reducing Factors")
                fig = plot_bar(negative_features, neg=True)
                tile.pyplot(fig)

            optimal_range_table = st.session_state.optimal_range_table
            optimal_range_table["Optimal Range - Lower Limit"] = optimal_range_table["optimal value"].apply(lambda x: x.left)
            optimal_range_table["Optimal Range - Upper Limit"] = optimal_range_table["optimal value"].apply(lambda x: x.right)
            optimal_range_table.drop("optimal value", axis=1, inplace=True)
            optimal_range_table["STATUS"] = optimal_range_table.apply(check_for_anom, axis=1)
            tile1.write(optimal_range_table)

def show_validation_results(cmatrix):
        tile = st.container(border = True)
        tn, fp, fn, tp = cmatrix

        tile = st.container(border=True)
        tile.markdown("<h4 style='text-align: center;'>Confusion Matrix</h4>", unsafe_allow_html=True)
        col1, col2, col3 = tile.columns([0.1, 0.8, 0.1])

        with col2:
            cl1, cl2 = st.columns(2)
            with cl1:
                tile = cl1.container(border=True)
                tile.metric("True Negative", tn)

                tile = cl1.container(border=True)
                tile.metric("False Negative", fn)
            
            with cl2:
                tile = cl2.container(border=True)
                tile.metric("False Positive", fp)

                tile = cl2.container(border=True)
                tile.metric("True Positive", tp)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            tile = col1.container(border=True)
            acc = (tp + tn)/(tp+fp+tn+fn) * 100
            tile.metric('Accuracy Score', round(acc, 2))

        with col2:
            tile = col2.container(border=True)

            pres = tp /(tp+fp) * 100
            rec = tp /(tp+fn) * 100
            f1 = 2 * (pres * rec)/(pres + rec)
            tile.metric('F1 Score', round(f1, 2))

        with col3:
            tile = col3.container(border=True)
            pres = tp /(tp+fp) * 100
            tile.metric('Precision Score', round(pres, 2))

        with col4:
            tile = col4.container(border=True)
            rec = tp /(tp+fn) * 100
            tile.metric('Recall Score', round(rec, 2))

st.markdown("<h1 style='text-align: center; '>Predictive Maintenance Platform</h1>", unsafe_allow_html=True)

container = st.container(border=True)
options = st.sidebar.radio("Pages", options=["Configuration", "Dashboard", "Analytics", "Validation", "Anomalies"])

if options == "Configuration":

    tile = st.container(border =True)
    tile.image("images/WhatsApp Image 2024-09-08 at 19.28.05.jpeg", caption="The equipment health curve shows three phases of maintenance: proactive (before potential failure), predictive (during deterioration), and reactive (post-failure). Predictive maintenance intervenes between potential and functional failure, reducing downtime by providing real-time monitoring and forecasts for timely repairs.")

    tile = st.container(border =True)
    tile.header(":toolbox: Predictive Maintenance")
    tile.write("Predictive Maintenance leverages sensor data and analytics to provide real-time monitoring and device failure predictions. This approach enhances reliability, minimizes downtime, and extends equipment lifespan by calculating risk scores and predicting failure probabilities over time, ensuring timely interventions.")
    
    tile = st.container(border=True)
    time = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    probability = [0, 0.30, 0.32, 0.40, 0.50, 0.60, 0.63, 0.65, 0.70, 0.85, 0.95, 0.95, 0.95, 0.97, 0.97
    , 0.98, 0.98, 0.98, 0.98, 0.98, 0.99]

    # Plot probability of failure over time
    fig, ax = plt.subplots()
    ax.plot(time, probability, label='Probability of Failure')
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Probability of Failure (%)')
    ax.set_title('Failure Rate Over Time')
    
    tile.pyplot(fig)
    
    tile.header(":firefighter: How It Helps!!")

    col1, col2 = tile.columns(2)

    with col1:
        col1.write("- Predict time to failure/lifetime\n- Prioritize maintenance tasks\n- Reduce maintenance checks\n- Focus on high-risk equipment")

    with col2:
        col2.write("- Reduces unexpected downtime\n- Extends equipment lifespan\n- Reduces costly emergency repairs.\n- Minimizes service disruptions")

    tile = st.container(border=True)
    tile.markdown("<h1 style='text-align: center; '>Key Inferences !!</h1>", unsafe_allow_html=True)

    col1, col2 = tile.columns(2)

    # Column 1: Customer Risk Levels
    with col1:
        tile = st.container(border=True)
        tile.header(":fire_engine: Customer Risk Levels")
        tile.write("Categorizes customers based on their current risk level. For example, a customer labeled as 'High Risk' has a greater than 80% chance of equipment failure within the next 90 days.")

    # Column 2: Probability of Failure Over Time
    with col2:
        tile = st.container(border=True)
        tile.header(":clock1: Probability of Failure Over Time")
        tile.write("A function representing the change in risk of failure over time. For example, a device's failure probability may start at 30% and rise to 70% over 60 days, indicating increasing risk.")

    configure_model()

if options == "Anomalies":
    data = st.session_state.data
    amd = AnomalyDetection(data)
    dat_, scores = amd.estimateKDE()
    fig, ax = plt.subplots()
    sns.lineplot(scores)
    ax.axhline(y=-19.03, linewidth=2, color='orange', ls='dotted', label="Alert Warning")
    ax.axhline(y=-20.00, linewidth=2, color='red', ls='dotted', label="Anomaly Warning")
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'lower center') 

    tile = st.container(border=True)
    tile.markdown("### Anomaly Detection")
    tile.markdown("Anomaly detection works by learning what normal sensor readings look like for a device and then setting limits to spot unusual behavior. It uses past data to understand how the device operates when everything is running smoothly. If new sensor readings fall outside these normal limits, an early warning is given. This helps identify problems early, allowing for quick action to prevent damage and reduce repair costs.")

    plt.title("Normal Operating Characteristics")
    plt.ylabel("Health Index")
    plt.xlabel("Time")

    tile = st.container(border=True)
    tile.pyplot(fig)
    tile.info("The Health Index is a single value that provides an overall summary of a device's health. It is calculated by analyzing data from all sensor readings, giving a quick indication of how well the device is functioning.")
    
    tile = st.container(border=True)
    tile.markdown("#### Individual Status")

    data = st.session_state.data

    devices = [None] + data["device"].unique().tolist()
    dev = tile.selectbox("Select a Device", devices)

    if dev != None:
        records = data[data["device"] == dev]
        records = records.groupby("device").last().reset_index()


        record__ = records.drop(["failure", "duration", "device"], axis=1)
        plot_bar_1(record__.melt())

        val = amd.predict_for_(records)
        tile = st.container(border=True)

        tile.markdown(f"<h1 style='text-align: center; color: red;'>{val}</h1>", unsafe_allow_html=True)

if options == "Dashboard":
    try:
        forge_dashboard()

        try:
            tile = st.container(border=True)
            tile.markdown("<h3 style='text-align: center; '>Lifetime Value of Machines</h3>", unsafe_allow_html=True)
            tile.markdown(f"<p style='text-align: center; '>Average Lifetime Value of a Device - {993.5} hrs</p>", unsafe_allow_html=True)
            data  = st.session_state.data
            data_1 = data[data["failure"] == 1]
            data_2 = data[data["failure"] == 0].sample(n=30)
            data = pd.concat([data_1, data_2])
            surv_func = AverageLifetime(data)
            fig, ax = plt.subplots()
            sns.lineplot(x=surv_func["time"], y=surv_func["prob_survival"])
            plt.axhline(y = .5, color="r", label="Median Point of Survival") 
            plt.xlabel("Time in Days")
            plt.ylabel("Probability of Survival")

            tile.pyplot(fig)

        except Exception as e:
            pass

        visualize_results()

    except Exception as e:
        tile = st.container(border=True)
        tile.markdown("<h1 style='text-align: center; color: red;'>Sorry, you need to train the model on the configuration page to view this page!!</h1>", unsafe_allow_html=True)

if options == "Analytics":

    try:
        with st.spinner("Setting-up Reasoning Platform"):

            expected_failure_times()

            data = st.session_state.data
            risk_map = st.session_state.risk_score

            opt_val = ExtractOptimumRange(data, risk_map)
            opt_val.set_optimum_ranges()

            st.session_state.optimum_val = opt_val.optimum_ranges

            provide_reasonings()

    except Exception as e:
        tile = st.container(border=True)
        tile.markdown("<h1 style='text-align: center; color: red;'>Sorry, you need to train the model on the configuration page to view this page!!</h1>", unsafe_allow_html=True)

if options =="Validation":
    try:
        tile = st.container(border = True)
        tile.markdown("#### Model Validation")
        tile.markdown("To assess model performance, data is split 'n' hours prior to failure to predict each device's expected failure time. The actual device status is observed within this n-hour window. If the model correctly predicts failure and the device fails, it is counted as a True Positive. If the device stays active and the model predicts a longer time to failure, it is a True Negative. Model performance is evaluated using metrics such as accuracy, F2 score, precision, and recall.")

        tab1, tab2, tab3 = st.tabs(["30hr Window", "15hr Window", "6hr Window"])

        with tab1:
            show_validation_results([10, 2, 28, 60])

        with tab2:
            show_validation_results([8, 2, 20, 70])


        with tab3:
            show_validation_results([25, 10, 15, 50])


        tile = st.container(border = True)
        tile.markdown("#### Concordance Index")
        tile.markdown("The Concordance Index (C-index) is a performance metric used in survival analysis to evaluate the predictive accuracy of a survival model. A pair is considered concordant if the individual (or device) with the higher predicted risk (i.e., shorter expected failure time) actually experiences the event before the one with the lower predicted risk (i.e., longer expected failure time). The C-index ranges from 0 to 1, where 1 indicates perfect concordance, and any value above 0.5 suggests better performance than random chance.")
        col1, col2, col3 = tile.columns([0.3, 0.4, 0.3])
        with col2:
            tile = col2.container(border=True)
            tile.metric("Model Concordance Index", round(st.session_state.cindex * 100, 2))
    except:
        tile = st.container(border=True)
        tile.markdown("<h1 style='text-align: center; color: red;'>Sorry, you need to train the model on the configuration page to view this page!!</h1>", unsafe_allow_html=True)

