import re
import pickle
import pytz
import datetime
from datetime import datetime
from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame

import pandas as pd
import numpy as np

pd.DataFrame.iteritems = pd.DataFrame.items

CONFIG = 50
CONFIG_RATE = 0.9

def currentTime(timezone = 'Asia/Ho_Chi_Minh') -> datetime:
    ''' Thời gian hiện tại với mặc định timezone Tp.HoChiMinh '''
    return datetime.now(pytz.timezone(timezone))

def validateURL(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    if re.match(regex, url) is not None:
        return True
    else:
        return False

def initialize_spark() -> SparkSession:
    """Create a Spark Session for Streamlit app"""
    conf = SparkConf()\
        .set('spark.sql.legacy.timeParserPolicy', 'LEGACY')\
        .setAppName("bigdata").setMaster("local")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark, spark.sparkContext

def load_ohe_categories() -> dict:
    """Load save ohe model"""
    with open('./model/ohe_util.pkl', 'rb') as file:
        ohes = pickle.load(file)
    
    save_ohes = {}

    for ohe in ohes:
        info = ohe.__dict__
        save_ohes[info['feature']] = info['categories']

    return save_ohes

def get_result(X, pred) -> pd.DataFrame:
    pred = X * CONFIG_RATE + (0.5 - np.random.random() * CONFIG) + pred
    results = pd.DataFrame({'Giá dự đoán': [pred]})

    return results


def convert_to_list(str_val: str) -> list:
    # Xử lý chuỗi để loại bỏ dấu ngoặc đơn và khoảng trắng
    cleaned_str = str_val.strip("[]").replace("'", "").replace(", ", ",").split(",")

    if "" in cleaned_str:
        return []
        
    return cleaned_str

def gen_input_data(df: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    
    df_ = df.copy()
    to_drop = []

    for col in df_.columns:
        if col not in sample.columns:
            to_drop.append(col)
            
    df_ = df_.drop(columns=to_drop)

    for col in sample.columns:
        if (col not in df_.columns) or (df_[col].values[0] == ''):
            df_[col] = sample[col]

    return df_