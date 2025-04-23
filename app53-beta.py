import streamlit as st
import pandas as pd
import numpy as np
from prophet.plot import plot_components_plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import datetime
from prophet import Prophet
from prophet.make_holidays import get_holiday_names
from sklearn.preprocessing import LabelEncoder
from statsmodels.graphics.tsaplots import plot_acf
from prophet.diagnostics import cross_validation, performance_metrics
import seaborn as sns
from itertools import product

st.title("Step-by-Step Time Series Forecasting")

# Initialize session state
if "selected_regressors" not in st.session_state:
    st.session_state["selected_regressors"] = []
if "external_regressors" not in st.session_state:
    st.session_state["external_regressors"] = []
if "add_all_regressors" not in st.session_state:
    st.session_state["add_all_regressors"] = False
if "use_external_regressors" not in st.session_state:
    st.session_state["use_external_regressors"] = False
if "use_grid_search" not in st.session_state:
    st.session_state["use_grid_search"] = False

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    # Step 2: Dataset Options
    st.header("1. Dataset")
    delimiter = st.radio("What is the separator?", [",", ";", "|"])
    date_format = st.selectbox("What is the date format?", ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y"])

    # Step 3: Column Selection
    st.subheader("Select Columns")
    df_temp = pd.read_csv(uploaded_file, delimiter=delimiter)
    uploaded_file.seek(0)
    date_column = st.selectbox("Select Date Column", df_temp.columns, key="date_col")
    target_options = [col for col in df_temp.columns if col != date_column]
    target_column = st.selectbox("Select Target Column", target_options, key="target_col") if target_options else None

    if not target_column:
        st.error("No valid target columns available. Please ensure your CSV has at least two columns.")
        st.stop()

    # Load data
    df = pd.read_csv(uploaded_file, delimiter=delimiter)
    df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='coerce')

    if df[date_column].isna().any():
        st.error("Some dates could not be parsed. Please check the date format.")
        st.stop()

    try:
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        if df[target_column].isna().all():
            st.error(f"Target column '{target_column}' contains no valid numeric values.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to convert target column '{target_column}' to numeric: {str(e)}")
        st.stop()

    initial_len = len(df)
    df = df.dropna(subset=[target_column])
    if len(df) < initial_len:
        st.warning(f"Dropped {initial_len - len(df)} rows with missing values in '{target_column}'.")

    # Interactive Data Explorer
    st.subheader("Raw Data")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if numeric_cols:
        num_cols = len(numeric_cols)
        fig_all = make_subplots(rows=num_cols, cols=1, subplot_titles=numeric_cols, shared_xaxes=True, vertical_spacing=0.1)
        for i, col in enumerate(numeric_cols, 1):
            fig_all.add_trace(go.Scatter(x=df[date_column], y=df[col], mode="lines", name=col), row=i, col=1)
        fig_all.update_layout(height=300 * num_cols, title_text="Raw Data: All Numeric Columns", showlegend=True)
        for i in range(1, num_cols + 1):
            fig_all.update_yaxes(title_text=numeric_cols[i-1], row=i, col=1)
        st.plotly_chart(fig_all, use_container_width=True)

    st.write("##### Processed Data Preview")
    st.write(df[[date_column, target_column]].head())

    # Step 4: Filtering Section
    st.subheader("Filtering Options")
    dimension_column = st.selectbox("Select Dataset Dimension (if any)", ["None"] + list(df.columns))
    selected_categories = []
    if dimension_column != "None":
        unique_values = df[dimension_column].unique().tolist()
        selected_categories = st.multiselect(f"Select categories from '{dimension_column}'", unique_values)
    aggregation_function = st.selectbox("Target Aggregation Function", ["sum", "mean", "min", "max"])
    if dimension_column != "None" and selected_categories:
        df_filtered = df[df[dimension_column].isin(selected_categories)]
    else:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        agg_dict = {target_column: aggregation_function}
        agg_dict.update({col: 'mean' for col in numeric_cols if col != target_column and col != date_column})
        df_filtered = df.groupby(date_column).agg(agg_dict).reset_index()
    st.write("##### Filtered Data Preview")
    st.write(df_filtered.head())

    # Step 5: Resampling Section
    st.subheader("Resampling Options")
    df_sorted = df_filtered.sort_values(by=date_column)
    df_sorted["time_diff"] = df_sorted[date_column].diff()
    detected_freq = pd.infer_freq(df_sorted[date_column]) or str(df_sorted["time_diff"].mode()[0])
    st.write(f"**Frequency detected in dataset:** {detected_freq}")
    resampling_freq = st.selectbox(
        "Select resampling frequency (if needed)",
        ["None", "T (Minute)", "H (Hourly)", "D (Daily)", "W (Weekly)", "M (Monthly)"]
    )
    if resampling_freq != "None":
        resampling_map = {"T (Minute)": "T", "H (Hourly)": "H", "D (Daily)": "D", "W (Weekly)": "W", "M (Monthly)": "M"}
        df_resampled = df_filtered.resample(resampling_map[resampling_freq], on=date_column).mean().reset_index()
    else:
        df_resampled = df_filtered.copy()
    st.write("##### Resampled Data Preview")
    st.write(df_resampled.head())

    # Step 6: Data Cleaning Section
    st.subheader("Data Cleaning Options")
    delete_zero = st.checkbox("Delete rows where target = 0")
    delete_negative = st.checkbox("Delete rows where target < 0")
    log_transform = st.checkbox("Apply log transformation to target (log<y+1>)")
    df_cleaned = df_resampled.copy()
    if delete_zero:
        df_cleaned = df_cleaned[df_cleaned[target_column] != 0]
    if delete_negative:
        df_cleaned = df_cleaned[df_cleaned[target_column] >= 0]
    if log_transform:
        df_cleaned[target_column] = df_cleaned[target_column].apply(lambda x: np.log(x + 1))
    for col in df_cleaned.columns:
        if col != date_column and df_cleaned[col].dtype in ["int64", "float64"]:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    st.write("##### Cleaned Data Preview")
    st.write(df_cleaned.head())
    st.session_state["prepared_data"] = df_cleaned.copy()
    st.session_state["date_column"] = date_column
    st.session_state["target_column"] = target_column

    # Step 7: Prior Scale Section
    st.header("2. Modeling")
    st.subheader("Prior Scale Settings")
    st.write("**Note**: Enable grid search for automatic tuning, or set parameters manually. High trend or seasonality prior scales may reduce regressor impact.")
    st.session_state["use_grid_search"] = st.checkbox("Use Grid Search for Automatic Parameter Tuning", value=False)
    if not st.session_state["use_grid_search"]:
        changepoint_prior_scale = st.number_input(
            "Trend Prior Scale",
            min_value=0.001, max_value=1.0, value=0.050, step=0.005,
            help="Controls trend flexibility. Higher values allow more changes."
        )
        seasonality_prior_scale = st.number_input(
            "Seasonality Prior Scale",
            min_value=1.0, max_value=50.0, value=10.0, step=1.0,
            help="Controls seasonality strength. Higher values allow more complex seasonality."
        )
        holidays_prior_scale = st.number_input(
            "Holidays Prior Scale",
            min_value=1.0, max_value=50.0, value=10.0, step=1.0,
            help="Controls holiday effects. Higher values increase holiday impact."
        )
        st.session_state["changepoint_prior_scale"] = changepoint_prior_scale
        st.session_state["seasonality_prior_scale"] = seasonality_prior_scale
        st.session_state["holidays_prior_scale"] = holidays_prior_scale
    regressor_prior_scale = st.number_input(
        "Regressor Prior Scale",
        min_value=0.1, max_value=50.0, value=10.0, step=1.0,
        help="Controls the flexibility of additional regressors. Higher values increase their influence."
    )
    st.session_state["regressor_prior_scale"] = regressor_prior_scale

    # Step 8: Seasonality Settings
    st.subheader("Seasonality Settings")
    if not st.session_state["use_grid_search"]:
        seasonality_mode = st.selectbox(
            "Seasonality Mode",
            options=["Additive", "Multiplicative"],
            help="Additive: Seasonality is added to the trend. Multiplicative: Seasonality is multiplied by the trend."
        )
        st.session_state["seasonality_mode"] = seasonality_mode.lower()
    else:
        st.write("Seasonality mode will be tuned automatically via grid search.")
    yearly_seasonality = st.radio(
        "Yearly Seasonality",
        options=["Auto", "False", "Custom"],
        help="Choose whether or not to include yearly seasonality."
    )
    monthly_seasonality = st.radio(
        "Monthly Seasonality",
        options=["Auto", "Custom"],
        help="Choose whether or not to include monthly seasonality."
    )
    weekly_seasonality = st.radio(
        "Weekly Seasonality",
        options=["Auto", "False", "Custom"],
        help="Choose whether or not to include weekly seasonality."
    )
    st.session_state["yearly_seasonality"] = yearly_seasonality
    st.session_state["monthly_seasonality"] = monthly_seasonality
    st.session_state["weekly_seasonality"] = weekly_seasonality

    # Step 9: Holidays Section
    st.subheader("Holiday Settings")
    country_list = ["None", "India", "USA", "France", "England", "Germany", "Canada", "Australia"]
    selected_country = st.selectbox("Select a country for holiday effects", country_list)
    use_public_holidays = st.checkbox("Include Public Holidays in the Model", value=True)
    st.session_state["selected_country"] = selected_country
    st.session_state["use_public_holidays"] = use_public_holidays

    # Step 10: Regressors Section
    st.subheader("Regressor Settings")
    available_regressors = [
        col for col in df_cleaned.select_dtypes(include=["int64", "float64"]).columns
        if col not in [date_column, target_column]
    ]
    if available_regressors:
        add_all_regressors = st.checkbox("Add All Detected Regressors", value=False)
        selected_regressors = []
        if not add_all_regressors:
            selected_regressors = st.multiselect("Select individual regressors to include", available_regressors)
        use_external_regressors = st.checkbox("Select External Regressor If Any", value=False)
        external_regressors = []
        if use_external_regressors:
            external_regressors = st.multiselect("Select external regressors from dataset", available_regressors)
        st.session_state["add_all_regressors"] = add_all_regressors
        st.session_state["selected_regressors"] = available_regressors if add_all_regressors else selected_regressors
        st.session_state["use_external_regressors"] = use_external_regressors
        st.session_state["external_regressors"] = external_regressors
        st.write("### Selected Regressors")
        if st.session_state["selected_regressors"] or st.session_state["external_regressors"]:
            selected_vars = st.session_state["selected_regressors"] + st.session_state["external_regressors"]
            st.write(f"**Regressors:** {', '.join(selected_vars)}")
        else:
            st.write("No regressors selected.")
    else:
        st.warning("No available numeric regressors found in the dataset.")

    # Step 11: Other Modeling Parameters
    st.subheader("Other Modeling Parameters")
    changepoint_range = st.number_input(
        "Changepoint Range",
        min_value=0.00, max_value=0.99, value=0.80, step=0.01
    )
    changepoints_input = st.text_area(
        "Specify Changepoint Dates (comma-separated, e.g., 2023-01-01,2023-06-01; leave empty for automatic)",
        value=""
    )
    if changepoints_input.strip():
        try:
            changepoints = [pd.to_datetime(date.strip(), format=date_format) for date in changepoints_input.split(",")]
            st.session_state["changepoints"] = changepoints
        except ValueError:
            st.error("Invalid date format in changepoints.")
            st.session_state["changepoints"] = None
    else:
        st.session_state["changepoints"] = None
    growth = st.selectbox("Growth", options=["Linear", "Logistic"])
    if growth.lower() == "logistic":
        floor = st.number_input("Logistic Growth Floor", value=0.0, help="Minimum value for logistic growth.")
        st.session_state["floor"] = floor
    st.session_state["changepoint_range"] = changepoint_range
    st.session_state["growth"] = growth
    st.session_state["date_format"] = date_format

    # Step 12: Preview Model Trend
    st.subheader("Model Trend Preview")
    if "prepared_data" not in st.session_state:
        st.error("Please process your data first.")
        st.stop()
    preview_df = st.session_state["prepared_data"].copy()
    preview_df = preview_df.rename(columns={st.session_state["date_column"]: "ds", st.session_state["target_column"]: "y"})
    if st.session_state["growth"].lower() == "logistic":
        preview_df['cap'] = preview_df['y'].max() * 1.1
        preview_df['floor'] = st.session_state.get("floor", 0.0)
    model_preview = Prophet(
        growth=st.session_state["growth"].lower(),
        changepoint_range=st.session_state["changepoint_range"],
        seasonality_mode=st.session_state.get("seasonality_mode", "additive"),
        changepoints=st.session_state.get("changepoints", None)
    )
    model_preview.fit(preview_df)
    future = model_preview.make_future_dataframe(periods=30)
    if st.session_state["growth"].lower() == "logistic":
        future['cap'] = preview_df['cap'].iloc[0]
        future['floor'] = preview_df['floor'].iloc[0]
    forecast = model_preview.predict(future)
    # Filter forecast to historical data only
    historical_forecast = forecast[forecast["ds"].isin(preview_df["ds"])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preview_df["ds"], y=preview_df["y"], mode="markers", name="Actual Data"))
    fig.add_trace(go.Scatter(x=historical_forecast["ds"], y=historical_forecast["yhat"], mode="lines", name="Forecasted Trend"))
    fig.update_layout(title="Impact of Changepoint Range & Growth on Forecast (Historical Data Only)", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig)

    # Step 13: Model Evaluation
    st.header("3. Model Evaluation")
    st.session_state["evaluate_model"] = st.checkbox("Evaluate my model")
    st.session_state["use_cross_validation"] = st.checkbox("Use Cross-Validation")
    if st.session_state["use_cross_validation"]:
        st.session_state["cv_folds"] = st.slider("Number of Cross-Validation Folds", min_value=1, max_value=10, value=5, step=1)
    if st.session_state["evaluate_model"]:
        df = st.session_state["prepared_data"]
        default_end_date = df[st.session_state["date_column"]].max().date()
        default_start_date = default_end_date - pd.Timedelta(days=365)
        col1, col2 = st.columns(2)
        with col1:
            val_start = st.date_input("Validation Start Date", value=default_start_date,
                                      min_value=df[st.session_state["date_column"]].min().date(),
                                      max_value=default_end_date)
        with col2:
            val_end = st.date_input("Validation End Date", value=default_end_date,
                                    min_value=val_start,
                                    max_value=df[st.session_state["date_column"]].max().date())
        val_start = pd.to_datetime(val_start)
        val_end = pd.to_datetime(val_end)
        st.session_state["val_start"] = val_start
        st.session_state["val_end"] = val_end
        total = len(df)
        val_data = df[(df[st.session_state["date_column"]] >= val_start) & (df[st.session_state["date_column"]] <= val_end)]
        percent = round((len(val_data) / total) * 100, 2)
        st.markdown(f"<div style='background-color:#f0f0f0;padding:10px'>Validation data: **{percent}%** of total</div>",
                    unsafe_allow_html=True)
    metric_options = ["MAPE", "RMSE", "SMAPE", "MAE", "MSE"]
    selected_metrics = st.multiselect("Choose one or more metrics to evaluate model performance:", metric_options)
    st.session_state["selected_metrics"] = selected_metrics

    # Step 14: Forecast
    st.header("4. Forecast")
    default_horizon = 10
    horizon_weeks = default_horizon
    custom_horizon = st.checkbox("Forecast horizon in weeks")
    if custom_horizon:
        horizon_weeks = st.number_input("How many weeks to forecast?", min_value=1, max_value=20, value=default_horizon, step=1)
    st.subheader("Future Regressor Values")
    st.write("**Note**: If using regressors, upload future values for accurate forecasts. Without this, regressors are set to 0 for future dates.")
    future_regressor_file = st.file_uploader("Upload CSV with future regressor values (optional)", type=["csv"])
    future_regressors_df = None
    if future_regressor_file is not None:
        future_regressors_df = pd.read_csv(future_regressor_file)
        future_regressors_df[date_column] = pd.to_datetime(future_regressors_df[date_column], format=date_format)
    if st.button("▶Run Forecast"):
        if "prepared_data" not in st.session_state:
            st.error("Please process your data first.")
            st.stop()
        df = st.session_state["prepared_data"]
        date_col = st.session_state["date_column"]
        target_col = st.session_state["target_column"]
        df_prophet = df.rename(columns={date_col: "ds", target_col: "y"})
        regressors = st.session_state.get("selected_regressors", []) + st.session_state.get("external_regressors", [])
        regressor_stats = {}
        for reg in regressors:
            if reg in df_prophet.columns:
                regressor_stats[reg] = {'mean': df_prophet[reg].mean(), 'std': df_prophet[reg].std()}
                df_prophet[reg] = (df_prophet[reg] - regressor_stats[reg]['mean']) / regressor_stats[reg]['std']
        def convert_seasonality_value(value):
            if value == "Auto":
                return "auto"
            elif value == "False":
                return False
            elif value == "Custom":
                return True
            return value
        # Grid Search
        best_params = None
        best_rmse = float('inf')
        if st.session_state.get("use_grid_search", False):
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.05, 0.5],
                'seasonality_prior_scale': [0.1, 1.0, 10.0],
                'holidays_prior_scale': [0.1, 1.0, 10.0],
                'seasonality_mode': ['additive'] if df_prophet['y'].le(0).any() else ['additive', 'multiplicative']
            }
            param_combinations = list(product(*param_grid.values()))
            total_combinations = len(param_combinations)
            st.write(f"Running grid search over {total_combinations} parameter combinations...")
            progress_bar = st.progress(0)
            for i, params in enumerate(param_combinations):
                param_dict = dict(zip(param_grid.keys(), params))
                model = Prophet(
                    changepoint_prior_scale=param_dict['changepoint_prior_scale'],
                    seasonality_prior_scale=param_dict['seasonality_prior_scale'],
                    holidays_prior_scale=param_dict['holidays_prior_scale'],
                    yearly_seasonality=convert_seasonality_value(st.session_state["yearly_seasonality"]),
                    weekly_seasonality=convert_seasonality_value(st.session_state["weekly_seasonality"]),
                    daily_seasonality=False,
                    changepoint_range=st.session_state["changepoint_range"],
                    growth=st.session_state["growth"].lower(),
                    seasonality_mode=param_dict['seasonality_mode'],
                    changepoints=st.session_state["changepoints"]
                )
                if st.session_state["monthly_seasonality"] == "Custom":
                    model.add_seasonality(name="monthly", period=30.44, fourier_order=5)
                if st.session_state["use_public_holidays"] and st.session_state["selected_country"] != "None":
                    model.add_country_holidays(country_name=st.session_state["selected_country"])
                for reg in regressors:
                    if reg in df_prophet.columns:
                        model.add_regressor(reg, prior_scale=st.session_state["regressor_prior_scale"], standardize=False)
                try:
                    model.fit(df_prophet)
                    # Adjust cross-validation parameters based on data length
                    data_days = (df_prophet['ds'].max() - df_prophet['ds'].min()).days
                    initial_days = max(365, int(data_days * 0.6))
                    df_cv = cross_validation(
                        model,
                        initial=f'{initial_days} days',
                        period='180 days',
                        horizon='365 days',
                        parallel="threads"
                    )
                    df_p = performance_metrics(df_cv)
                    rmse = df_p['rmse'].mean()
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = param_dict
                except Exception as e:
                    st.warning(f"Parameter combination {param_dict} failed: {str(e)}")
                    continue
                progress_bar.progress((i + 1) / total_combinations)
            progress_bar.empty()
            if best_params:
                st.success(f"Grid search completed! Best parameters: {best_params}, RMSE: {best_rmse:.2f}")
                st.session_state["changepoint_prior_scale"] = best_params['changepoint_prior_scale']
                st.session_state["seasonality_prior_scale"] = best_params['seasonality_prior_scale']
                st.session_state["holidays_prior_scale"] = best_params['holidays_prior_scale']
                st.session_state["seasonality_mode"] = best_params['seasonality_mode']
            else:
                st.error("Grid search failed for all combinations. Using default parameters.")
                st.session_state["changepoint_prior_scale"] = 0.05
                st.session_state["seasonality_prior_scale"] = 10.0
                st.session_state["holidays_prior_scale"] = 10.0
                st.session_state["seasonality_mode"] = "additive"
        # Train final model
        model = Prophet(
            changepoint_prior_scale=st.session_state["changepoint_prior_scale"],
            seasonality_prior_scale=st.session_state["seasonality_prior_scale"],
            holidays_prior_scale=st.session_state["holidays_prior_scale"],
            yearly_seasonality=convert_seasonality_value(st.session_state["yearly_seasonality"]),
            weekly_seasonality=convert_seasonality_value(st.session_state["weekly_seasonality"]),
            daily_seasonality=False,
            changepoint_range=st.session_state["changepoint_range"],
            growth=st.session_state["growth"].lower(),
            seasonality_mode=st.session_state["seasonality_mode"],
            changepoints=st.session_state["changepoints"]
        )
        if st.session_state["monthly_seasonality"] == "Custom":
            model.add_seasonality(name="monthly", period=30.44, fourier_order=5)
        if st.session_state["use_public_holidays"] and st.session_state["selected_country"] != "None":
            model.add_country_holidays(country_name=st.session_state["selected_country"])
        for reg in regressors:
            if reg in df_prophet.columns:
                model.add_regressor(reg, prior_scale=st.session_state["regressor_prior_scale"], standardize=False)
        with st.spinner("Training final model and generating forecast..."):
            model.fit(df_prophet)
        st.session_state["trained_model"] = model
        st.write(f"Model trained with regressors: {', '.join([reg for reg in regressors if reg in df_prophet.columns])}")
        future_periods = horizon_weeks * 7
        future_df = model.make_future_dataframe(periods=future_periods, freq="D")
        for reg in regressors:
            if reg in df_prophet.columns:
                if future_regressors_df is not None and reg in future_regressors_df.columns:
                    future_df = future_df.merge(future_regressors_df[[date_column, reg]], left_on='ds', right_on=date_column, how='left')
                    future_df[reg] = (future_df[reg] - regressor_stats[reg]['mean']) / regressor_stats[reg]['std']
                    future_df[reg] = future_df[reg].fillna(0)
                    future_df = future_df.drop(columns=[date_column])
                else:
                    future_df[reg] = 0
        if st.session_state["growth"].lower() == "logistic":
            df_prophet['cap'] = df_prophet['y'].max() * 1.1
            df_prophet['floor'] = st.session_state.get("floor", 0.0)
            future_df['cap'] = df_prophet['cap'].iloc[0]
            future_df['floor'] = df_prophet['floor'].iloc[0]
        forecast = model.predict(future_df)
        st.session_state["forecast_result"] = forecast
        st.success(f"Forecast generated for {horizon_weeks} weeks!")
        if st.session_state.get("evaluate_model") and "selected_metrics" in st.session_state and st.session_state["selected_metrics"]:
            metric_values = {}
            # Holdout validation
            val_start = st.session_state["val_start"]
            val_end = st.session_state["val_end"]
            actual_val = df[(df[date_col] >= val_start) & (df[date_col] <= val_end)]
            forecast_val = forecast[(forecast["ds"] >= val_start) & (forecast["ds"] <= val_end)]
            if not actual_val.empty and not forecast_val.empty:
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                evaluation_df = forecast_val[["ds", "yhat"]].merge(
                    actual_val[[date_col, target_col]],
                    left_on="ds",
                    right_on=date_col,
                    how="inner"
                )
                y_true = evaluation_df[target_col].values
                y_pred = evaluation_df["yhat"].values
                if "MAPE" in st.session_state["selected_metrics"]:
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                    metric_values["Holdout_MAPE"] = round(mape, 2)
                if "RMSE" in st.session_state["selected_metrics"]:
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    metric_values["Holdout_RMSE"] = round(rmse, 2)
                if "SMAPE" in st.session_state["selected_metrics"]:
                    smape = 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
                    metric_values["Holdout_SMAPE"] = round(smape, 2)
                if "MAE" in st.session_state["selected_metrics"]:
                    mae = mean_absolute_error(y_true, y_pred)
                    metric_values["Holdout_MAE"] = round(mae, 2)
                if "MSE" in st.session_state["selected_metrics"]:
                    mse = mean_squared_error(y_true, y_pred)
                    metric_values["Holdout_MSE"] = round(mse, 2)
            else:
                st.error(f"No data in validation range {val_start} to {val_end}.")
            # Cross-validation
            if st.session_state.get("use_cross_validation", False):
                try:
                    df_cv = cross_validation(
                        model,
                        initial='730 days',
                        period='180 days',
                        horizon='365 days',
                        parallel="threads"
                    )
                    df_p = performance_metrics(df_cv)
                    for metric in st.session_state["selected_metrics"]:
                        metric_lower = metric.lower()
                        if metric_lower in df_p.columns:
                            metric_values[f"CV_{metric}"] = round(df_p[metric_lower].mean(), 2)
                except Exception as e:
                    st.warning(f"Cross-validation failed: {str(e)}")
            st.subheader("Evaluation Metrics")
            st.write("Evaluation based on selected validation dates and/or cross-validation")
            st.json(metric_values)
        merged = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
            df[[date_col, target_col]],
            left_on="ds",
            right_on=date_col,
            how="left"
        )
        st.subheader("Forecast Overview")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged["ds"], y=merged[target_col], mode="markers", name="Actual", marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=merged["ds"], y=merged["yhat"], mode="lines", name="Forecast", line=dict(width=2)))
        fig.add_trace(go.Scatter(
            x=pd.concat([merged["ds"], merged["ds"][::-1]]),
            y=pd.concat([merged["yhat_upper"], merged["yhat_lower"][::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name="Uncertainty Interval",
            showlegend=True
        ))
        fig.update_layout(
            title="Forecast vs. Actual with Uncertainty Intervals",
            xaxis_title="Date",
            yaxis_title=target_col,
            legend_title="Data Series",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Forecast Components")
        comp_fig = plot_components_plotly(model, forecast)
        st.plotly_chart(comp_fig, use_container_width=True)
        st.subheader("Residuals (Prediction Errors)")
        merged["residual"] = merged[target_col] - merged["yhat"]
        residual_fig = px.line(merged, x="ds", y="residual", title="Residuals Over Time")
        residual_fig.update_layout(xaxis_title="Date", yaxis_title="Residual")
        st.plotly_chart(residual_fig, use_container_width=True)
        st.subheader("Anomalies in Historical Data")
        historical = forecast[forecast["ds"].isin(df[date_col])]
        merged_anomaly = historical[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
            df[[date_col, target_col]], left_on="ds", right_on=date_col, how="inner"
        )
        merged_anomaly["anomaly"] = (merged_anomaly[target_col] < merged_anomaly["yhat_lower"]) | (merged_anomaly[target_col] > merged_anomaly["yhat_upper"])
        fig_anomaly = px.scatter(
            merged_anomaly, x="ds", y=target_col, color="anomaly",
            title="Anomalies (Red = Outlier)",
            color_discrete_map={True: "red", False: "blue"}
        )
        fig_anomaly.add_traces([
            go.Scatter(x=merged_anomaly["ds"], y=merged_anomaly["yhat"], name="Forecast", mode="lines"),
            go.Scatter(x=merged_anomaly["ds"], y=merged_anomaly["yhat_lower"], name="Lower Bound", mode="lines", line=dict(dash="dot")),
            go.Scatter(x=merged_anomaly["ds"], y=merged_anomaly["yhat_upper"], name="Upper Bound", mode="lines", line=dict(dash="dot")),
        ])
        fig_anomaly.update_layout(legend_title="Data Series")
        st.plotly_chart(fig_anomaly, use_container_width=True)
        st.subheader("Additional Visual Insights")
        with st.expander("Interactive Forecast Table"):
            st.write("Explore forecast data with dates, predictions, and confidence intervals.")
            forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
                columns={"ds": "Date", "yhat": "Prediction", "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"}
            )
            st.dataframe(forecast_table, use_container_width=True)
        with st.expander("Changepoints"):
            st.write("Vertical lines indicate changepoints where the trend changes.")
            historical_merged = merged[merged["ds"].isin(df[date_col])]
            fig_changepoints = go.Figure()
            fig_changepoints.add_trace(go.Scatter(x=historical_merged["ds"], y=historical_merged["yhat"], mode="lines", name="Forecast"))
            fig_changepoints.add_trace(go.Scatter(x=historical_merged["ds"], y=historical_merged[target_col], mode="markers", name="Actual"))
            changepoints = model.changepoints if st.session_state["changepoints"] is None else st.session_state["changepoints"]
            for cp in changepoints:
                fig_changepoints.add_vline(x=cp, line_dash="dash", line_color="red")
            fig_changepoints.update_layout(
                title="Forecast with Changepoints (Historical Data Only)",
                xaxis_title="Date",
                yaxis_title=target_col,
                showlegend=True
            )
            st.plotly_chart(fig_changepoints, use_container_width=True)
        if st.session_state["use_public_holidays"] and st.session_state["selected_country"] != "None":
            with st.expander("Holiday Effects"):
                st.write("Impact of holidays on the forecast.")
                fig_holidays = px.line(forecast, x="ds", y="holidays", title="Holiday Effects Over Time")
                fig_holidays.update_layout(xaxis_title="Date", yaxis_title="Holiday Effect")
                st.plotly_chart(fig_holidays, use_container_width=True)
        with st.expander("Target vs. Features Relationships"):
            st.write("Scatter plots showing relationships between the target and other numeric features.")
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            if numeric_cols and len(numeric_cols) > 1:
                for col in numeric_cols:
                    if col != target_col and col != date_col and col in df.columns:
                        correlation = df[[col, target_col]].corr().iloc[0, 1]
                        st.write(f"Correlation between {target_col} and {col}: {correlation:.2f}")
                        if abs(correlation) < 0.1:
                            st.warning(f"Low correlation ({correlation:.2f}) between {target_col} and {col}. Consider excluding this regressor.")
                        fig = px.scatter(df, x=col, y=target_col, title=f"{target_col} vs. {col}",
                                         labels={col: col, target_col: target_col})
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available for scatter plots.")
        if numeric_cols and len(numeric_cols) > 1:
            with st.expander("Correlation Heatmap"):
                st.write("Correlation between numeric features, including the target.")
                corr = df[numeric_cols].corr()
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title="Correlation Heatmap of Numeric Features"
                )
                st.plotly_chart(fig, use_container_width=True)
        with st.expander("Component Weightage"):
            st.write("Quantifies the impact of trend, seasonality, and holidays.")
            components = ["trend", "yearly", "weekly"]
            if st.session_state["use_public_holidays"] and st.session_state["selected_country"] != "None":
                components.append("holidays")
            std_values = []
            for comp in components:
                if comp in forecast:
                    std = forecast[comp].std()
                    std_values.append(std)
                else:
                    std_values.append(0)
            for comp, std in zip(components, std_values):
                st.write(f"Standard Deviation of {comp.capitalize()}: {std:.2f}")
            fig = px.bar(
                x=components,
                y=std_values,
                title="Component Weightage (Standard Deviation)",
                labels={"x": "Component", "y": "Standard Deviation"}
            )
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Forecast vs. Actual Scatter"):
            st.write("Scatter plot of predicted vs. actual values.")
            valid_data = merged.dropna(subset=[target_col])
            fig = px.scatter(
                valid_data, x="yhat", y=target_col,
                title="Predicted vs. Actual Values",
                labels={"yhat": "Predicted", target_col: "Actual"}
            )
            fig.add_trace(go.Scatter(
                x=[valid_data["yhat"].min(), valid_data["yhat"].max()],
                y=[valid_data["yhat"].min(), valid_data["yhat"].max()],
                mode="lines",
                name="Ideal Line",
                line=dict(dash="dash")
            ))
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Seasonal Heatmap"):
            st.write("Heatmap of average target values by day of week and month.")
            df_heatmap = df.copy()
            df_heatmap["day_of_week"] = df_heatmap[date_col].dt.day_name()
            df_heatmap["month"] = df_heatmap[date_col].dt.month_name()
            pivot = df_heatmap.pivot_table(
                values=target_col,
                index="day_of_week",
                columns="month",
                aggfunc="mean"
            )
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            months_order = ["January", "February", "March", "April", "May", "June",
                            "July", "August", "September", "October", "November", "December"]
            pivot = pivot.reindex(days_order).reindex(columns=months_order)
            fig = px.imshow(
                pivot,
                text_auto=True,
                color_continuous_scale="Viridis",
                title="Average Target by Day of Week and Month"
            )
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Autocorrelation of Residuals"):
            st.write("Autocorrelation function (ACF) of residuals.")
            plt.figure(figsize=(10, 4))
            plot_acf(merged["residual"].dropna(), lags=40)
            st.pyplot(plt)
        if st.session_state.get("use_cross_validation", False):
            with st.expander("Cross-Validation Results"):
                st.write("RMSE across forecast horizons for cross-validation folds.")
                st.write("**Note**: Lower RMSE indicates better accuracy. A sharp increase suggests the model is less reliable for longer forecasts.")
                try:
                    df_cv = cross_validation(
                        model,
                        initial='730 days',
                        period='180 days',
                        horizon='365 days',
                        parallel="threads"
                    )
                    df_p = performance_metrics(df_cv)
                    fig = px.line(
                        df_p, x="horizon", y="rmse",
                        title="Cross-Validation RMSE by Forecast Horizon",
                        labels={"horizon": "Forecast Horizon", "rmse": "RMSE"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Cross-validation visualization failed: {str(e)}")
        with st.expander("Regressor Importance"):
            st.write("Impact of additional regressors on the forecast (based on coefficients).")
            if regressors and any(reg in df_prophet.columns for reg in regressors) and hasattr(model, 'extra_regressors'):
                coefficients = []
                valid_regressors = []
                for reg in regressors:
                    if reg in df_prophet.columns and reg in model.extra_regressors:
                        reg_index = list(model.extra_regressors.keys()).index(reg)
                        coef = model.params['beta'][0][reg_index]
                        coefficients.append(coef)
                        valid_regressors.append(reg)
                if valid_regressors:
                    fig = px.bar(x=valid_regressors, y=coefficients, title="Regressor Coefficients",
                                 labels={"x": "Regressor", "y": "Coefficient"})
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("**Note**: Non-zero coefficients indicate regressor impact. Small values may suggest low predictive power or scaling issues.")
                else:
                    st.warning("No valid regressors were included in the model. Check if selected regressors exist in the dataset.")
            else:
                st.warning("No regressors selected, available in the dataset, or included in the model.")
        with st.expander("Lag Plot"):
            st.write("Plots target at time t vs. t-k to show temporal dependencies. Strong patterns suggest past values predict future ones, aiding model tuning.")
            lags = [1, 7, 30]
            fig = make_subplots(rows=1, cols=len(lags), subplot_titles=[f"Lag {lag}" for lag in lags])
            for i, lag in enumerate(lags, 1):
                lag_data = pd.DataFrame({
                    f"t-{lag}": df[target_col].shift(lag),
                    "t": df[target_col]
                }).dropna()
                fig.add_trace(go.Scatter(x=lag_data[f"t-{lag}"], y=lag_data["t"], mode="markers", name=f"Lag {lag}"),
                              row=1, col=i)
            fig.update_layout(title="Lag Plots", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
