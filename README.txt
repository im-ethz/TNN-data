How to get the data:

0. find login information in config.py

1. scrape TrainingPeaks using scrape_trainingpeaks.py (select a date-range and workout-type bike)
	a) for each athlete, move the training files to a subdirectory with their name in the "raw" folder
	b) make sure everything is within the right time-range and remove duplicates by searching for (1) (2), etc.

2. convert TrainingPeaks fit.gz files to csv using the bike2csv library
	a) extract every .fit.gz file to .fit
	b) convert fit file to csv

3. preprocess dexcom and trainingpeaks
	a) first steps of dexcom and trainingpeaks can be run at the same time
	b) once you reach timezone preprocessing, make sure dexcom and trainingpeaks are at this point in the code, so all preprocessed data from dexcom and trainingpeaks can be used to obtain the final timezones
