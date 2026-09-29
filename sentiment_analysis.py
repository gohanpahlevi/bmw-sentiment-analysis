"""
Employee sentiment on flexible working in the German automotive industry.

Spark implementation of the analysis originally written in pandas for an
Advanced Project Study at TUM. Reads German-language employer reviews that have
already been labelled for sentiment toward home office, flexibility, flextime
and mobile working, then reports how that sentiment moved across the pre-COVID,
COVID and post-COVID periods.

Usage:
    python sentiment_analysis.py
"""

import os
import shutil
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
)

DATA = os.environ.get("DATA_DIR", "data")
OUT = os.environ.get("OUT_DIR", "output")
RESULTS = os.environ.get("RESULTS_DIR", "results")

CARMAKERS = ["BMW", "Mercedes-Benz", "Audi", "Porsche", "Volkswagen"]
MIN_REVIEWS = 100
FIRST_YEAR = 2015


AUTOMOTIVE_SCHEMA = StructType([
    StructField("Company", StringType()),
    StructField("Year", IntegerType()),
    StructField("Role", StringType()),
    StructField("Standardized Role (DE)", StringType()),
    StructField("Standardized Role (EN)", StringType()),
    StructField("Rating", DoubleType()),
    StructField("Source", StringType()),
    StructField("NACE Code", StringType()),
    StructField("Review Text", StringType()),
    StructField("Sentiment", StringType()),
    StructField("Main Topic", StringType()),
    StructField("Standardized Topic (DE)", StringType()),
    StructField("Standardized Topic (EN)", StringType()),
])

NON_AUTOMOTIVE_SCHEMA = StructType([
    StructField("Company", StringType()),
    StructField("Year", IntegerType()),
    StructField("Role", StringType()),
    StructField("Standardized Role (DE)", StringType()),
    StructField("Standardized Role (EN)", StringType()),
    StructField("Rating", DoubleType()),
    StructField("Source", StringType()),
    StructField("Review Text", StringType()),
    StructField("Sentiment", StringType()),
])


def build_session():
    return (
        SparkSession.builder
        .appName("automotive-sentiment")
        .master("local[*]")
        # 200 is the default and far too many partitions for this volume.
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def read_reviews(spark):
    """Load both review sets and stack them with a dataset label."""
    # Review text contains line breaks, so multiLine is required. Without it
    # rows split on those breaks and every count downstream is wrong.
    opts = dict(header=True, multiLine=True, escape='"', encoding="UTF-8")

    automotive = (
        spark.read.schema(AUTOMOTIVE_SCHEMA)
        .csv(os.path.join(DATA, "Automotive_Final.csv"), **opts)
        .withColumn("dataset", F.lit("automotive"))
    )

    # The comparison set has no topic columns, hence allowMissingColumns.
    non_automotive = (
        spark.read.schema(NON_AUTOMOTIVE_SCHEMA)
        .csv(os.path.join(DATA, "Non_Automotive_Final.csv"), **opts)
        .withColumn("dataset", F.lit("non_automotive"))
    )

    return automotive.unionByName(non_automotive, allowMissingColumns=True)


def clean(df):
    """Score sentiment, assign periods, drop rows we cannot use."""
    # The labelling prompt was German, so both spellings appear.
    label = F.lower(F.trim(F.col("Sentiment")))
    score = (
        F.when(label.isin("positive", "positiv"), F.lit(1))
        .when(label.isin("negative", "negativ"), F.lit(-1))
        .when(label == "neutral", F.lit(0))
        .otherwise(F.lit(None).cast(IntegerType()))
    )

    period = (
        F.when(F.col("Year").between(2015, 2019), F.lit("1_pre_covid"))
        .when(F.col("Year").between(2020, 2022), F.lit("2_covid"))
        .when(F.col("Year") >= 2023, F.lit("3_post_covid"))
        .otherwise(F.lit(None).cast(StringType()))
    )

    return (
        df
        .withColumn("sentiment_score", score)
        .withColumn("period", period)
        .filter(F.col("Year") >= FIRST_YEAR)
        .filter(F.col("sentiment_score").isNotNull())
    )


def drop_thin_companies(df):
    """Keep only companies with at least MIN_REVIEWS reviews."""
    # A window avoids collecting the counts to the driver and filtering twice.
    return (
        df
        .withColumn("n", F.count("*").over(Window.partitionBy("Company")))
        .filter(F.col("n") >= MIN_REVIEWS)
        .drop("n")
    )


def sentiment_by_company_period(df):
    return (
        df.groupBy("dataset", "Company", "period")
        .agg(
            F.count("*").alias("n_reviews"),
            F.round(F.avg("sentiment_score"), 3).alias("mean_sentiment"),
            F.sum(F.when(F.col("sentiment_score") == 1, 1).otherwise(0)).alias("n_positive"),
            F.sum(F.when(F.col("sentiment_score") == -1, 1).otherwise(0)).alias("n_negative"),
            F.round(F.avg("Rating"), 2).alias("mean_rating"),
        )
        .withColumn(
            "pct_negative",
            F.round(100 * F.col("n_negative") / F.col("n_reviews"), 1),
        )
        .orderBy("dataset", "Company", "period")
    )


def rating_by_year(df):
    """Average overall rating per carmaker per year."""
    # Passing the value list keeps the pivot from scanning to discover it.
    return (
        df.filter(F.col("Company").isin(CARMAKERS))
        .filter(F.col("Year").between(2018, 2024))
        .groupBy("Year")
        .pivot("Company", CARMAKERS)
        .agg(F.round(F.avg("Rating"), 2))
        .orderBy("Year")
    )


def top_topics_per_year(df, n=5):
    """The n most mentioned topics in each year."""
    counts = (
        df.filter(F.col("Standardized Topic (EN)").isNotNull())
        .filter(F.col("Year").between(2020, 2024))
        .groupBy("Year", "Standardized Topic (EN)")
        .agg(F.count("*").alias("mentions"))
    )

    ranked = Window.partitionBy("Year").orderBy(F.desc("mentions"))
    return (
        counts
        .withColumn("rank", F.row_number().over(ranked))
        .filter(F.col("rank") <= n)
        .orderBy("Year", "rank")
    )


def sentiment_by_role(df, min_total=50):
    return (
        df.filter(F.col("Standardized Role (EN)").isNotNull())
        .groupBy("Standardized Role (EN)")
        .agg(
            F.count("*").alias("total"),
            F.round(100 * F.avg(F.when(F.col("sentiment_score") == 1, 1.0).otherwise(0.0)), 1).alias("pct_positive"),
            F.round(100 * F.avg(F.when(F.col("sentiment_score") == -1, 1.0).otherwise(0.0)), 1).alias("pct_negative"),
        )
        .filter(F.col("total") >= min_total)
        .orderBy(F.desc("pct_negative"))
    )


def bmw_against_sector(df):
    """BMW's mean sentiment per period against the automotive mean."""
    sector = (
        df.filter(F.col("dataset") == "automotive")
        .groupBy("period")
        .agg(F.avg("sentiment_score").alias("sector_mean"))
    )

    bmw = (
        df.filter(F.col("Company") == "BMW")
        .groupBy("period")
        .agg(
            F.count("*").alias("bmw_n"),
            F.avg("sentiment_score").alias("bmw_mean"),
        )
    )

    # Three rows, so broadcast it and skip the shuffle.
    return (
        bmw.join(F.broadcast(sector), on="period", how="left")
        .withColumn("gap", F.round(F.col("bmw_mean") - F.col("sector_mean"), 3))
        .withColumn("bmw_mean", F.round("bmw_mean", 3))
        .withColumn("sector_mean", F.round("sector_mean", 3))
        .orderBy("period")
    )


def validation_sample(df, per_class=67, seed=42):
    """Stratified sample for manual spot checking of the sentiment labels."""
    # sampleBy takes fractions, so counts come out approximate.
    sizes = {
        r["Sentiment"]: r["n"]
        for r in df.groupBy("Sentiment").agg(F.count("*").alias("n")).collect()
    }
    fractions = {k: min(1.0, per_class / v) for k, v in sizes.items() if v}
    return df.sampleBy("Sentiment", fractions=fractions, seed=seed)


def write_csv(df, name):
    """Write a summary table as one CSV file rather than a directory of parts."""
    tmp = os.path.join(RESULTS, f".{name}")
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp)
    part = next(f for f in os.listdir(tmp) if f.endswith(".csv"))
    os.replace(os.path.join(tmp, part), os.path.join(RESULTS, f"{name}.csv"))
    shutil.rmtree(tmp)


def show(title, df, n=25):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    df.show(n, truncate=40)


def main():
    spark = build_session()
    spark.sparkContext.setLogLevel("ERROR")

    reviews = drop_thin_companies(clean(read_reviews(spark)))
    # Seven actions read this, so keep it in memory rather than replaying.
    reviews.cache()

    print(f"\nreviews after filtering: {reviews.count()}")
    print(f"companies: {reviews.select('Company').distinct().count()}")

    tables = {
        "sentiment_by_company_period": sentiment_by_company_period(reviews),
        "rating_by_year": rating_by_year(reviews),
        "top_topics_per_year": top_topics_per_year(reviews),
        "sentiment_by_role": sentiment_by_role(reviews),
        "bmw_against_sector": bmw_against_sector(reviews),
    }

    show("Sentiment by company and period", tables["sentiment_by_company_period"], 40)
    show("Average rating by year", tables["rating_by_year"])
    show("Top topics per year", tables["top_topics_per_year"], 30)
    show("Sentiment by role", tables["sentiment_by_role"])
    show("BMW against the automotive sector", tables["bmw_against_sector"])

    os.makedirs(RESULTS, exist_ok=True)
    for name, table in tables.items():
        write_csv(table, name)
    print(f"\nresult tables written to {RESULTS}")

    print(f"\nvalidation sample: {validation_sample(reviews).count()} rows")

    (
        reviews.write.mode("overwrite")
        .partitionBy("dataset", "period")
        .parquet(os.path.join(OUT, "reviews"))
    )
    print(f"parquet written to {OUT}")

    reviews.unpersist()
    spark.stop()


if __name__ == "__main__":
    if not os.path.isdir(DATA):
        sys.exit(f"No data directory at {DATA}. See README.")
    main()
