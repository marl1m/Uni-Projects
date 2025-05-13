
from sklearn.preprocessing import RobustScaler

def preprocess_data(df):
    df = df.fillna(method="ffill")
    scaler = RobustScaler()
    return scaler.fit_transform(df)
