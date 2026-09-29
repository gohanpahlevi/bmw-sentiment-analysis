import numpy as np
import pytest
from pyspark.sql import SparkSession

from sentiment_analysis import (
    bootstrap_ci,
    bootstrap_diff_p,
    clean,
    drop_thin_companies,
)


@pytest.fixture(scope="session")
def spark():
    s = (
        SparkSession.builder
        .appName("tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield s
    s.stop()


def make(spark, rows):
    return spark.createDataFrame(rows, "Company string, Year int, Sentiment string")


def test_sentiment_accepts_both_languages(spark):
    df = make(spark, [
        ("BMW", 2021, "positive"), ("BMW", 2021, "positiv"),
        ("BMW", 2021, "negative"), ("BMW", 2021, "negativ"),
        ("BMW", 2021, "neutral"),
    ])
    scores = sorted(r["sentiment_score"] for r in clean(df).collect())
    assert scores == [-1, -1, 0, 1, 1]


def test_unlabelled_rows_are_dropped(spark):
    df = make(spark, [("BMW", 2021, "positive"), ("BMW", 2021, "unklar")])
    assert clean(df).count() == 1


def test_sentiment_is_case_and_space_insensitive(spark):
    df = make(spark, [("BMW", 2021, "  Positive "), ("BMW", 2021, "NEGATIV")])
    scores = sorted(r["sentiment_score"] for r in clean(df).collect())
    assert scores == [-1, 1]


@pytest.mark.parametrize("year,expected", [
    (2015, "1_pre_covid"),
    (2019, "1_pre_covid"),
    (2020, "2_covid"),
    (2022, "2_covid"),
    (2023, "3_post_covid"),
    (2024, "3_post_covid"),
])
def test_period_boundaries(spark, year, expected):
    df = make(spark, [("BMW", year, "neutral")])
    assert clean(df).collect()[0]["period"] == expected


def test_years_before_2015_are_dropped(spark):
    df = make(spark, [("BMW", 2014, "neutral"), ("BMW", 2015, "neutral")])
    assert clean(df).count() == 1


def test_thin_companies_are_dropped(spark):
    rows = [("BMW", 2021, "neutral")] * 120 + [("Tiny", 2021, "neutral")] * 5
    kept = drop_thin_companies(clean(make(spark, rows)))
    assert {r["Company"] for r in kept.select("Company").distinct().collect()} == {"BMW"}


def test_bootstrap_interval_brackets_the_mean():
    values = np.concatenate([np.ones(60), np.zeros(40)])
    lo, hi = bootstrap_ci(values)
    assert lo < values.mean() < hi


def test_identical_samples_are_not_significant():
    a = np.array([1, 0, -1] * 40, dtype=float)
    assert bootstrap_diff_p(a, a.copy()) > 0.5


def test_clearly_different_samples_are_significant():
    a = np.ones(100)
    b = -np.ones(100)
    assert bootstrap_diff_p(a, b) < 0.01
