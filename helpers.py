def calculate_non_percentage(df):
    """
    Calculate percentage of null values in a spark dataframe. 
    """
    df_nan_count = df.select(
    [psf.count(psf.when(psf.col(c).isNull(), True)).alias(c)
     for c in df.columns])
    return df_nan_count.toPandas() / df.count()
 
