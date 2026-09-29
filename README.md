# Employee sentiment on home office and flexible working in the German automotive industry

A Spark job over German-language employer reviews, looking at how sentiment toward home office, flexibility, flextime and mobile working changed at BMW and its competitors between 2015 and 2024.

## Question

Since the end of the pandemic a lot of German employers have pushed to bring people back to the office. What each company announces is public. What its own employees think of it is not, and there is no reason to assume every carmaker landed in the same place. This reads what employees wrote and checks whether BMW moved differently from the rest of the industry.

## Data

Employer reviews from four platforms, collected and labelled in two stages.

**Collection.** Scraped with Selenium, walking the pagination for each company and writing one file per employer. Every review available at the time of the scrape was taken, so the counts below are the full set for each company and not a draw from it.

| Source | Automotive | Non-automotive |
| --- | --- | --- |
| Kununu | 2,905 | 1,235 |
| Glassdoor | 731 | 2,269 |
| Stepstone | 134 | 0 |
| Indeed | 17 | 0 |

**Labelling.** Each review was scored by a LLaMA 3.1 model running locally through Ollama. The prompt is German and rates the review only on its tone toward home office, flexibility, flextime and mobile working, reading both explicit statements such as "Homeoffice erlaubt" and implicit signals such as ten hour days, missing breaks and contactability outside working time. Running the model locally rather than through a hosted API meant no review text left the machine.

| Set | Reviews | Companies | Used as |
| --- | --- | --- | --- |
| Automotive | 3,787 | 17 | Main sample |
| Non-automotive | 3,504 | 13 | Comparison |

Companies with fewer than 100 reviews are dropped, which leaves 19 companies and 6,352 reviews from 2015 onward.

The raw review text is not in this repository and neither are the scrapers. The reviews belong to the platforms, and a German employer review carries role, employer and year next to the free text, which at a small company is enough to work out who wrote it. So neither goes in a public repository. `sentiment_analysis.py` expects the two CSV files in `data/`, with the schemas declared at the top of the script, and the aggregated output is committed under `results/` so every figure below can be checked without them.

## Results

BMW started above the industry average and ended below it.

| Period | Reviews | Mean sentiment | Negative |
| --- | --- | --- | --- |
| Pre-COVID 2015-2019 | 287 | +0.042 | 19.2% |
| COVID 2020-2022 | 212 | 0.000 | 26.9% |
| Post-COVID 2023-2024 | 202 | -0.139 | 35.1% |

Mercedes-Benz over the same three periods sat at +0.048, +0.019 and +0.055, with negative sentiment flat between 23% and 25%.

Against the automotive mean, BMW was 0.084 above it before COVID and 0.067 below it after.

## How much of that survives a significance test

Bootstrap intervals on the mean, 2,000 resamples, 95%:

| Company | Period | n | Mean | 95% interval |
| --- | --- | --- | --- | --- |
| BMW | Pre-COVID | 287 | +0.042 | -0.031 to +0.115 |
| BMW | COVID | 212 | 0.000 | -0.099 to +0.099 |
| BMW | Post-COVID | 202 | -0.139 | -0.238 to -0.035 |
| Mercedes-Benz | Post-COVID | 238 | +0.055 | -0.034 to +0.147 |

BMW's post-COVID interval is the only one of the three that excludes zero. So what holds up is that BMW's post-COVID score is genuinely negative. The pre-COVID score sits too close to zero to call it anything.

Comparing companies against each other is where it gets thinner. Across all ten pairs of carmakers from 2020 onward, the closest to significant is BMW against Mercedes-Benz at p = 0.033. But ten pairs means ten chances to find something by luck, so the bar has to move. Correcting for that puts BMW against Mercedes-Benz at 0.33, and no pair in the set clears it.

So the claim that holds is that BMW's sentiment fell while Mercedes-Benz stayed flat. The claim that does not hold is that BMW fell by significantly more than its competitors.

Audi and Porsche are worth a line of their own. Both were already clearly negative before COVID, at -0.182 and -0.275, and both intervals exclude zero. BMW's did not. So BMW's post-COVID number is a change in position, while for those two it was the position they started from.

Two more things hold across the whole sample. Working hours is the most mentioned topic in every year from 2020 to 2024, and remote work moved from fourth place in 2020 to third in 2023 and 2024. Former employees are the most negative group, at 38.6% negative against 20.7% for interns.

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
pip install -r requirements-dev.txt
pytest
```

`validation_sample()` writes a stratified sample of labelled reviews to `output/validation_sample/` for manual spot checking. That file holds raw review text, so it stays under `output/`, which is gitignored, and never under `results/`.

## Notes on the implementation

`multiLine=True` on the CSV reader is not optional. Review text contains line breaks, and without it Spark splits rows on those breaks. It is the difference between 3,787 rows and about 6,000.

Schemas are declared rather than inferred, so the files are read once instead of twice.

The 100-review filter uses a window rather than collecting counts to the driver. The topic ranking uses `row_number()` over a window partitioned by year. The BMW comparison broadcasts the three-row sector table instead of shuffling the full dataset. There are no Python UDFs; everything is built-in `pyspark.sql.functions`, which keeps it inside the optimiser.

`spark.sql.shuffle.partitions` is set to 8. The default of 200 makes a job this size slower than pandas.

## Limitations

Reviews are voluntary, so the people who write them are not a random sample of employees. Every available review was collected, so nothing was lost by sampling, but people write these reviews when they have something to say, and the pool leans toward strong opinions and toward people who have already left.

The role breakdown shows the effect directly. Former employees are the most negative group at 38.6% negative across 580 reviews, against interns at 20.7%. Any company-level number carries that composition with it.

The sentiment labels come from a model, not from human annotation. `validation_sample()` draws a stratified sample so they can be checked by hand. It uses `sampleBy`, which works on fractions, so the sample size comes out approximate rather than exact.

Sentiment is measured only on home office and flexible working. A company can score badly here and well overall.

Sample sizes per company and period run from 51 to 378, which is why the intervals are as wide as they are. Audi and Porsche have the fewest reviews per period and so the widest intervals, and any year-on-year movement in those two should be read with that in mind.
