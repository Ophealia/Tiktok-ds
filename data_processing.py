from site import makepath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    df = pd.read_csv(filepath, index_col=0)
    df.reset_index(drop=True, inplace=True)
    return df

def clean_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def feature_engineering(df, save_path=None):
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    df["date"] = df["time"].dt.date
    df["dayofweek"] = df["time"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
    
    def categorize_hour(hour):
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 24:
            return "evening"
        else:
            return "night"
        
    df["time_period"] = df["hour"].apply(categorize_hour)
    df["video_category_num"] = df["video_category"].astype("category").cat.codes
    df["time_period_num"] = df["time_period"].astype("category").cat.codes
    
    df["total_action"] = df["like_type"] + df["relay_type"]
    
    user_stats = df.groupby("user_id").agg(
        total_views=("video_id", "count"), 
        like_ratio=("like_type", "mean") 
    ).reset_index()

    df["user_category_like_ratio"] = df.groupby(["user_id", "video_category_num"])["like_type"].transform("mean")

    
    video_stats = df.groupby("video_id").agg(
        total_watched=("user_id", "count"), 
        like_ratio=("like_type", "mean"), 
        share_ratio=("relay_type", "mean") 
    ).reset_index()
    
    df = df.merge(user_stats, on="user_id", how="left")
    df = df.merge(video_stats, on="video_id", how="left")
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df

if __name__ == "__main__":
    
    filepath = "data/dy_action.csv"
    df = load_data(filepath)
    df = clean_data(df)
    
    df = feature_engineering(df, save_path="data/data_process.csv")