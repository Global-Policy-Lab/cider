import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit


def all_spark(df):
    df = (df
          .withColumn('weekday', F.when(F.dayofweek('day').isin([1, 7]), 'weekend').otherwise('weekday'))
          .withColumn('daytime', F.when((F.hour('timestamp') < 7)|(F.hour('timestamp') >= 19), 'night').otherwise('day'))
          .withColumn('direction', lit('out'))
          .withColumn('directions', F.array(lit('in'), col('direction')))
          .withColumn('direction', F.explode('directions'))
          .withColumn('caller_id_copy', col('caller_id'))
          .withColumn('caller_id', F.when(col('direction') == 'in', col('recipient_id')).otherwise(col('caller_id')))
          .withColumn('recipient_id', F.when(col('direction') == 'in', col('caller_id_copy')).otherwise(col('recipient_id')))
          .drop('directions', 'caller_id_copy'))

    return active_days(df)


def active_days(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    out = (df
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.countDistinct('day').alias('active_days')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['active_days'])

    col_selection = [col(col_name).alias('active_days_' + col_name) for col_name in out.columns if col_name != 'caller_id']
    out = out.select('caller_id', *col_selection)

    return out


def add_all_cat(df, col_mapping):
    for column, value in col_mapping.items():
        df = (df
              .withColumn(column, F.array(lit(value), col(column)))
              .withColumn(column, F.explode(column)))

    return df


def pivot_df(df, index, columns, values):
    while columns:
        column = columns.pop()
        df = (df
              .groupby(index + columns)
              .pivot(column)
              .agg(*[F.first(val).alias(val) for val in values]))
        values = [val for val in df.columns if val not in index and val not in columns]

    return df
