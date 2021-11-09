import pyspark
import pyspark.sql.functions as psf
import pyspark.sql.window as psw


def confusion_matrix_curve(
        df: pyspark.sql.DataFrame,
        score_col: str,
        label_col: str
        ) -> pyspark.sql.DataFrame:
    """Calculates true, false positives, and true, false negatives for different thresholds.
    Values are calculated for all unique values in score_col with label_col containing
    the true labels.
    Note: The implementation is resticted to the binary classification task.
    Parameters
    ----------
    df : pyspark.sql.DataFrame
    score_col : str
        Name of the column containing the score output of the model.
        Higher score means higher probability of positive class.
    label_col : str
        Name of the columns containing the true label with values.
        Values must be in {0, 1} where 1 represents the positive class.
    Returns
    -------
    pyspark.sql.DataFrame
        Dataframe with columns score_col, true_positives, false_positives,
        true_negatives, false_negatives
    """
    base = df.groupby(score_col).agg(
        psf.sum(label_col).alias('threshold_true_positives'),
        psf.count(label_col).alias('threshold_counts'),
    )
    window = psw.Window.orderBy(
        psf.col(score_col).desc()
    ).rowsBetween(psw.Window.unboundedPreceding, 0)

    res = base.select([
        psf.col(score_col),
        psf.sum(psf.col('threshold_true_positives')).over(window).alias('true_positives'),
        psf.sum(psf.col('threshold_counts')).over(window).alias('positives')
    ])

    total_positives = res.select(psf.max(psf.col('true_positives'))).collect()[0][0]
    total_negatives = df.count() - total_positives
    res = res.select([
        '*',
        (psf.col('positives') - psf.col('true_positives')).alias('false_positives'),
        (psf.lit(total_positives) - psf.col('true_positives')).alias('false_negatives'),
    ])

    return res.withColumn(
        'true_negatives',
        psf.lit(total_negatives) - psf.col('false_positives')
    )


def precision_recall_curve(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """Adds precision and recall curve for different thresholds to the input dataframe.
    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe containing confusin matrix for different thresholds.
    Returns
    -------
    pyspark.sql.DataFrame
    """
    return df.select([
        '*',
        (psf.col('true_positives') / (psf.col('true_positives') + psf.col('false_negatives'))).alias('recall'),
        (psf.col('true_positives') / (psf.col('true_positives') + psf.col('false_positives'))).alias('precision'),
    ])