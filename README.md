# Employee sentiment on flexible working in the German automotive industry

Spark analysis of German-language employer reviews, looking at how sentiment toward home office, flexibility, flextime and mobile working changed at BMW and its competitors between 2015 and 2024.

Originally written in pandas for an Advanced Project Study at TUM School of Management, then reimplemented in PySpark.

## Question

German carmakers spent the COVID years rewriting how and where people work. The question is whether employees noticed, and whether the answer differs between manufacturers.

## Data

Employer reviews from public review sites, labelled for sentiment by a locally hosted LLaMA 3.1 running through Ollama. The prompt is German and scores each review only on its tone toward flexible working, reading both explicit statements and implicit signals such as long hours, missing breaks and contactability outside working time.

| Set | Reviews | Companies | Used as |
| --- | --- | --- | --- |
| Automotive | 3,787 | 17 | Main sample |
| Non-automotive | 3,504 | 13 | Comparison |

Companies with fewer than 100 reviews are dropped, which leaves 19 companies and 6,352 reviews from 2015 onward.

The raw review text is not in this repository. `sentiment_analysis.py` expects the two CSV files in `data/`, with the schemas declared at the top of the script. The aggregated output is committed under `results/`, so every figure below can be checked without it.

## Results

BMW moved from better than its sector to worse than it.

| Period | Reviews | Mean sentiment | Negative |
| --- | --- | --- | --- |
| Pre-COVID 2015-2019 | 287 | +0.042 | 19.2% |
| COVID 2020-2022 | 212 | 0.000 | 26.9% |
| Post-COVID 2023-2024 | 202 | -0.139 | 35.1% |

Mercedes-Benz over the same three periods sat at +0.048, +0.019 and +0.055, with negative sentiment flat between 23% and 25%.

Against the automotive mean, BMW was 0.084 above it before COVID and 0.067 below it after.

## How much of that survives a significance test

Less than the table above suggests, and it is worth being explicit about.

Bootstrap intervals on the mean, 2,000 resamples, 95%:

| Company | Period | n | Mean | 95% interval |
| --- | --- | --- | --- | --- |
| BMW | Pre-COVID | 287 | +0.042 | -0.031 to +0.115 |
| BMW | COVID | 212 | 0.000 | -0.099 to +0.099 |
| BMW | Post-COVID | 202 | -0.139 | -0.238 to -0.035 |
| Mercedes-Benz | Post-COVID | 238 | +0.055 | -0.034 to +0.147 |

BMW's post-COVID interval is the only one of the three that excludes zero. So the defensible claim is that BMW's sentiment on flexible working is negative after COVID and was not distinguishable from neutral before it.

The between-company claim is weaker. Comparing all ten carmaker pairs on 2020 onward, BMW against Mercedes-Benz gives p = 0.033, which is the smallest in the set. Ten comparisons means a Bonferroni factor of ten, and the corrected value is 0.33. Nothing in the matrix is significant after correction.

That does not make the descriptive trend uninteresting, and the direction is consistent across three periods and two independent measures. It does mean the honest statement is that BMW declined against a flat comparator, not that BMW declined significantly more than its competitors.

Two other findings hold across the sample. Working hours is the most mentioned topic in every year from 2020 to 2024, with remote work climbing from fourth to third. And by role, former employees are the most negative at 38.6% against interns at 20.7%.

Sentiment is scored +1, 0 or -1, so the mean is bounded at plus and minus one.

## Running it

Spark needs a JVM.

```
brew install openjdk@17
export JAVA_HOME=/opt/homebrew/opt/openjdk@17

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python sentiment_analysis.py
```

Summary tables go to `results/` as CSV and the full labelled dataset to `output/` as parquet, partitioned by dataset and period.

```
results/
  sentiment_by_company_period.csv   19 companies across three periods
  rating_by_year.csv                average overall rating, five carmakers
  top_topics_per_year.csv           five most mentioned topics per year
  sentiment_by_role.csv             sentiment split by job level
  bmw_against_sector.csv            BMW against the automotive mean
  company_intervals.csv             bootstrap intervals per company and period
  company_comparisons.csv           pairwise tests, raw and corrected
```

`sample_data/` holds a small generated set with the same schema, so the script runs end to end without the real reviews:

```
DATA_DIR=sample_data python sentiment_analysis.py
```

Tests:

```
pytest
```

## Notes on the implementation

`multiLine=True` on the CSV reader is not optional. Review text contains line breaks, and without it Spark splits rows on those breaks. It is the difference between 3,787 rows and about 6,000.

Schemas are declared rather than inferred, so the files are read once instead of twice.

The 100-review filter uses a window rather than collecting counts to the driver. The topic ranking uses `row_number()` over a window partitioned by year. The BMW comparison broadcasts the three-row sector table instead of shuffling the full dataset. There are no Python UDFs; everything is built-in `pyspark.sql.functions`, which keeps it inside the optimiser.

`spark.sql.shuffle.partitions` is set to 8. The default of 200 makes a job this size slower than pandas.

## Limitations

Reviews are self-selected, so people with something to say are overrepresented, and former employees more so than current ones.

The sentiment labels come from a model, not from human annotation. `validation_sample()` draws a stratified sample for manual spot checking. It uses `sampleBy`, which works on fractions, so the sample size is approximate rather than exact.

Sentiment is measured only on flexible working. A company can score badly here and well overall.

Sample sizes per company and period run from 51 to 378, which is why the intervals are as wide as they are. Audi and Porsche in particular have too few reviews per period to say much.
