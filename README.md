# TNN-data: Preprocessing of Team Novo Nordisk (TNN) cycling and diabetes data

This repository was created to give insights into the preprocessing of the Team Novo Nordisk (TNN) cycling and diabetes data. This dataset was used in the following manuscript: "Glycemic Patterns of Male Professional Athletes With Type 1 Diabetes During Exercise, Recovery and Sleep: Retrospective, Observational Study Over an Entire Competitive Season" by E. van Weenen et al.

This dataset is not publicly available to protect the privacy of the participants, but can be requested in limited form from the corresponding author upon reasonable request. The code for the analysis of this paper can be found under https://github.com/im-ethz/TNN-analysis.

## Data access and processing
Below you will find the steps used to access and process the data.

0. Find login information in `credentials.yml`

1. Scrape TrainingPeaks using `scrape_trainingpeaks.py` (select a date-range and workout-type bike)

2. For each athlete, structure the downloaded `.fit`/`.tcx`/`.pwx` files as follows. Make sure all files are in a folder called `raw`, and within raw, make subdirectories for each participant, e.g., `0`, `1`, etc

3. Convert TrainingPeaks `fit.gz` files to `csv` using the [bike2csv](https://github.com/evavanweenen/bike2csv) library by first extracting every `.fit.gz` file to `.fit` and subsequently converting `.fit` files to `.cvs`

4. Scrape Dexcom using `scrape_dexcom.py`

4. Preprocess dexcom and trainingpeaks using `preprocess_dexcom.py` and `preprocess_trainingpeaks.py`. Note that once you reach timezone processing for both, things get a little tricky and may require both trainingpeaks and dexcom data to be at a certain stage of the preprocessing.
