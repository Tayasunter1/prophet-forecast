from prophet import Prophet

def train_prophet(df, features=[], yearly=True, weekly=True, daily=False):
    model = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly, daily_seasonality=daily)

    for feature in features:
        model.add_regressor(feature)

    model.fit(df)
    future = model.make_future_dataframe(periods=30)

    for feature in features:
        future[feature] = df[feature].iloc[-1]  # Assuming static values for now

    forecast = model.predict(future)
    return model, forecast