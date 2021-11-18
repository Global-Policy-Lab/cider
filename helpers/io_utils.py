from box import Box
import geopandas as gpd  # type: ignore[import]
from geopandas import GeoDataFrame
from helpers.utils import get_spark_session
from pandas import DataFrame as PandasDataFrame
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, date_trunc, to_timestamp
from typing import Dict, List, Optional, Union


def load_generic(cfg: Box,
                 fname: Optional[str] = None,
                 df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None, 
                 location: str = None) -> SparkDataFrame:
    """
    Args:
        cfg: box object containing config data
        fname: path to file or folder with files
        df: pandas or spark df, if already loaded

    Returns: loaded spark df
    """
    spark = get_spark_session(cfg)

    if location == 'file':
        fname = 'file:' + fname
    if location == 'hdfs':
        fname = 'hdfs://localhost:9000' + fname
    
    # Load from file
    if fname is not None:
        # Load data if in a single file
        if '.csv' in fname:
            df = spark.read.csv(fname, header=True)

        # Load data if in chunks
        else:
            df = spark.read.csv(fname + '/*.csv', header=True)

    # Load from pandas dataframe
    if df is not None:
        if not isinstance(df, SparkDataFrame):
            df = spark.createDataFrame(df)

    # Issue with filename/dataframe provided
    else:
        raise ValueError('No filename or pandas/spark dataframe provided.')

    return df


def check_cols(df: Union[GeoDataFrame, PandasDataFrame, SparkDataFrame],
               required_cols: List[str],
               error_msg: str) -> None:
    """
    Check that the df has all required columns

    Args:
        df: spark df
        required_cols: columns that are required
        error_msg: error message to print if some columns are missing
    """
    if not set(required_cols).intersection(set(df.columns)) == set(required_cols):
        raise ValueError(error_msg)


def check_colvalues(df: SparkDataFrame, colname: str, colvalues: list, error_msg: str) -> None:
    """
    Check that a column has all required values

    Args:
        df: spark df
        colname: column to check
        colvalues: requires values
        error_msg: error message to print if values don't match
    """
    if set(df.select(colname).distinct().rdd.map(lambda r: r[0]).collect()).union(set(colvalues)) != set(colvalues):
        raise ValueError(error_msg)


def standardize_col_names(df: SparkDataFrame, col_names: Dict[str, str]) -> SparkDataFrame:
    """
    Rename columns, as specified in config file, to standard format

    Args:
        df: spark df
        col_names: mapping between standard column names and existing ones

    Returns: spark df with standardized column names

    """
    col_mapping = {v: k for k, v in col_names.items()}

    for col in df.columns:
        df = df.withColumnRenamed(col, col_mapping[col])

    return df


def load_cdr(cfg: Box,
             fname: Optional[str] = None,
             df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
             verify: bool = True) -> SparkDataFrame:
    """
    Load CDR data into spark df

    Args:
        cfg: box object containing config data
        fname: path to file or folder with files
        df: pandas or spark df, if already loaded
        verify: whether to check if right columns and values are present

    Returns: spark df
    """
    spark = get_spark_session(cfg)
    # load data as generic df and standardize column_names
    if df is not None:
        if not isinstance(df, SparkDataFrame):
            cdr = spark.createDataFrame(df)
        else:
            cdr = df
    elif fname is not None:
        # TODO: modularize file location behavior
        cdr = load_generic(cfg, fname=fname, df=df)
    else:
        raise ValueError('No filename or pandas/spark dataframe provided.')
    cdr = standardize_col_names(cdr, cfg.col_names.cdr)

    if verify:
        # Check that required columns are present
        required_cols = ['txn_type', 'caller_id', 'recipient_id', 'timestamp', 'duration', 'international']
        error_msg = 'CDR format incorrect. CDR must include the following columns: ' + ', '.join(required_cols)
        check_cols(cdr, required_cols, error_msg)

        # Check txn_type column
        error_msg = 'CDR format incorrect. Column txn_type can only include call and text.'
        check_colvalues(cdr, 'txn_type', ['call', 'text'], error_msg)

        # Clean international column
        error_msg = 'CDR format incorrect. Column international can only include domestic, international, and other.'
        check_colvalues(cdr, 'international', ['domestic', 'international', 'other'], error_msg)

    # Clean timestamp column
    cdr = cdr.withColumn('timestamp', to_timestamp(cdr['timestamp'], 'yyyy-MM-dd HH:mm:ss')) \
        .withColumn('day', date_trunc('day', col('timestamp')))

    # Clean duration column
    cdr = cdr.withColumn('duration', col('duration').cast('float'))

    return cdr


def load_antennas(cfg: Box,
                  fname: Optional[str] = None,
                  df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
                  verify: bool = True) -> SparkDataFrame:
    """
    Load antennas' dataset, and print % of antennas that are missing coordinates

    Args:
        cfg: box object containing config data
        fname: path to file
        df: pandas or spark df, if already loaded
        verify: whether to check if right columns and values are present

    Returns: spark df
    """
    spark = get_spark_session(cfg)
    # load data as generic df and standardize column_names
    if df is not None:
        if not isinstance(df, SparkDataFrame):
            antennas = spark.createDataFrame(df)
        else:
            antennas = df
    elif fname is not None:
        antennas = load_generic(cfg, fname=fname, df=df)
    else:
        raise ValueError('No filename or pandas/spark dataframe provided.')
    antennas = standardize_col_names(antennas, cfg.col_names.antennas)

    if verify:
        required_cols = ['antenna_id', 'latitude', 'longitude']
        error_msg = 'Antenna format incorrect. Antenna dataset must include the following columns: ' + ', '.join(
            required_cols)
        check_cols(antennas, required_cols, error_msg)

        antennas = antennas.withColumn('latitude', col('latitude').cast('float')).withColumn('longitude',
                                                                                             col('longitude').cast(
                                                                                                 'float'))
        print('Warning: %i antennas missing location' % (
                antennas.count() - antennas.select(['latitude', 'longitude']).na.drop().count()))

    return antennas


def load_recharges(cfg: Box,
                   fname: Optional[str] = None,
                   df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None) -> SparkDataFrame:
    """
    Load recharges' dataset

    Args:
        cfg: box object containing config data
        fname: path to file or folder with files
        df: pandas or spark df, if already loaded

    Returns: spark df
    """
    spark = get_spark_session(cfg)
    # load data as generic df and standardize column_names
    if df is not None:
        if not isinstance(df, SparkDataFrame):
            recharges = spark.createDataFrame(df)
        else:
            recharges = df
    elif fname is not None:
        recharges = load_generic(cfg, fname=fname, df=df)
    else:
        raise ValueError('No filename or pandas/spark dataframe provided.')
    recharges = standardize_col_names(recharges, cfg.col_names.recharges)

    # Clean timestamp column
    recharges = recharges.withColumn('timestamp', to_timestamp(recharges['timestamp'], 'yyyy-MM-dd HH:mm:ss')) \
        .withColumn('day', date_trunc('day', col('timestamp')))

    # Clean duration column
    recharges = recharges.withColumn('amount', col('amount').cast('float'))

    return recharges


def load_mobiledata(cfg: Box,
                    fname: Optional[str] = None,
                    df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None) -> SparkDataFrame:
    """
    Load mobile data dataset

    Args:
        cfg: box object containing config data
        fname: path to file or folder with files
        df: pandas or spark df, if already loaded

    Returns: spark df
    """
    spark = get_spark_session(cfg)
    # load data as generic df and standardize column_names
    if fname is not None:
        mobiledata = load_generic(cfg, fname=fname, df=df)
    elif df is not None:
        if not isinstance(df, SparkDataFrame):
            mobiledata = spark.createDataFrame(df)
        else:
            mobiledata = df
    else:
        raise ValueError('No filename or pandas/spark dataframe provided.')
    mobiledata = standardize_col_names(mobiledata, cfg.col_names.mobiledata)

    # Clean timestamp column
    mobiledata = mobiledata.withColumn('timestamp', to_timestamp(mobiledata['timestamp'], 'yyyy-MM-dd HH:mm:ss')) \
        .withColumn('day', date_trunc('day', col('timestamp')))

    # Clean duration column
    mobiledata = mobiledata.withColumn('volume', col('volume').cast('float'))

    return mobiledata


def load_mobilemoney(cfg: Box,
                     fname: Optional[str] = None,
                     df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
                     verify: bool = True) -> SparkDataFrame:
    """
    Load mobile money dataset

    Args:
        cfg: box object containing config data
        fname: path to file or folder with files
        df: pandas or spark df, if already loaded
        verify: whether to check if right columns and values are present

    Returns: spark df
    """
    spark = get_spark_session(cfg)
    # load data as generic df and standardize column_names
    if fname is not None:
        mobilemoney = load_generic(cfg, fname=fname, df=df)
    elif df is not None:
        if not isinstance(df, SparkDataFrame):
            mobilemoney = spark.createDataFrame(df)
        else:
            mobilemoney = df
    else:
        raise ValueError('No filename or pandas/spark dataframe provided.')
    mobilemoney = standardize_col_names(mobilemoney, cfg.col_names.mobilemoney)

    if verify:
        # Check that required columns are present
        required_cols = ['txn_type', 'caller_id', 'recipient_id', 'timestamp', 'amount']
        error_msg = 'Mobile money format incorrect. Mobile money records must include the following columns: ' + \
                    ', '.join(required_cols)
        check_cols(mobilemoney, required_cols, error_msg)

        # Check txn_type column
        txn_types = ['cashin', 'cashout', 'p2p', 'billpay', 'other']
        error_msg = 'Mobile money format incorrect. Column txn_type can only include ' + ', '.join(txn_types)
        check_colvalues(mobilemoney, 'txn_type', txn_types, error_msg)

    # Clean timestamp column
    mobilemoney = mobilemoney.withColumn('timestamp', to_timestamp(mobilemoney['timestamp'], 'yyyy-MM-dd HH:mm:ss')) \
        .withColumn('day', date_trunc('day', col('timestamp')))

    # Clean duration column
    mobilemoney = mobilemoney.withColumn('amount', col('amount').cast('float'))

    # Clean balance columns
    for c in mobilemoney.columns:
        if 'balance' in c:
            mobilemoney = mobilemoney.withColumn(c, col(c).cast('float'))

    return mobilemoney


def load_shapefile(fname: str) -> GeoDataFrame:
    """
    Load shapefile and make sure it has the right columns

    Args:
        fname: path to file, which can be .shp or .geojson

    Returns: GeoDataFrame

    """
    shapefile = gpd.read_file(fname)

    # Verify that columns are correct
    required_cols = ['region', 'geometry']
    error_msg = 'Shapefile format incorrect. Shapefile must include the following columns: ' + ', '.join(required_cols)
    check_cols(shapefile, required_cols, error_msg)

    shapefile['region'] = shapefile['region'].astype(str)

    return shapefile
