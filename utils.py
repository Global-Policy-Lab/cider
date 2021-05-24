import sys
import os
import datetime
import json
import pandas as pd
import shutil
import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, Window

def get_spark_session():
    '''
    Gets or creates spark session, with context and logging preferences set
    '''
    # Build spark session
    spark = SparkSession \
        .builder \
        .appName("mm") \
        .getOrCreate()
    # Change logging to just error messages
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def save_df(df, outfname, sep=','):
    ''' 
    Saves spark dataframe to csv file, using work-around to deal with spark's automatic partitioning and naming
    '''
    outfolder = outfname[:-4]
    df.repartition(1).write.csv(path=outfolder, mode="overwrite", header="true", sep=sep)
    # Work around to deal with spark automatic naming
    old_fname = [fname for fname in os.listdir(outfolder) if fname[-4:] == '.csv'][0]
    os.rename(outfolder + '/' + old_fname, outfname)
    shutil.rmtree(outfolder)

def save_parquet(df, outfname):
    '''
    Save spark dataframe to parquet file
    '''
    df.write.mode('overwrite').parquet(outfname)

def filter_dates_dataframe(df, start_date, end_date, colname='timestamp'):
    if colname not in df.columns:
        raise ValueError('Cannot filter dates because missing timestamp column')
    df = df.where(col(colname) >= pd.to_datetime(start_date))
    df = df.where(col(colname) < pd.to_datetime(end_date) + pd.Timedelta(days=1))
    return df

def make_dir(fname):
    if os.path.isdir(fname):
        shutil.rmtree(fname)
    os.mkdir(fname)