"""
Scrape Trainingpeaks
Select by:
- workout type = bike
- date range = 01/01/2020 until 31/10/2020 -> 2019
"""
from config import credentials, rider_mapping
import pyderman as dr
from selenium import webdriver
from tqdm import tqdm

import pandas as pd 
import time

download_path = '/local/home/evanweenen/hype-data/data/TrainingPeaks/export/'
dates = pd.date_range(start='01-01-2014', end='31-12-2021', freq='30D')

url_login = 'https://home.trainingpeaks.com/login'

driver_path = dr.install(browser=dr.chrome, file_directory='./lib/', verbose=True, chmod=True, overwrite=False, version='86.0.4240.22', filename=None, return_info=False)
options = webdriver.ChromeOptions()
options.add_experimental_option("prefs", {"download.default_directory":download_path})
driver = webdriver.Chrome(driver_path, options=options)
driver.maximize_window()

driver.get(url_login)

# login
driver.find_element_by_name('Username').send_keys(credentials['TP']['pro']['username'])
driver.find_element_by_name('Password').send_keys(credentials['TP']['pro']['password'])
driver.find_element_by_name('submit').click()
time.sleep(15)

driver.find_element_by_class_name('calendar').click()
time.sleep(5)

# select athlete
driver.find_element_by_class_name('groupAndAthleteSelector').click() # dropdown menu
time.sleep(5)
athletes = driver.find_elements_by_class_name('athleteOption') # list of athletes

retry = True

for i in range(len(athletes)):
	print(athletes[i].text)
	if ' '.join(athletes[i].text.split()[1:]).lower() in rider_mapping.keys() or \
		athletes[i].text.lower() == 'kusztor peter':
		print("include")
		athletes[i].click() # click on ith athlete
		time.sleep(5)
		
		# go to list layout
		driver.find_element_by_class_name('workoutSearch').click()
		time.sleep(5)

		# select only bike training (only first time visiting website)
		if i == 0 or retry:
			driver.find_element_by_class_name('filter').click()
			driver.set_page_load_timeout(10)

			driver.find_element_by_xpath("//*[@id='main']/div[1]/div/div/div[3]/div[3]/div/div[2]/div[5]/div[4]/div[2]/label[2]").click() # select bike
			driver.set_page_load_timeout(10)

		for d in tqdm(range(len(dates)-1)):
			start_date = dates[d]
			end_date = dates[d+1]

			driver.find_element_by_class_name('endDate').clear()
			driver.find_element_by_class_name('endDate').send_keys(end_date.strftime('%m/%d/%Y')+'\n')
			time.sleep(5)

			driver.find_element_by_class_name('startDate').clear()
			driver.find_element_by_class_name('startDate').send_keys(start_date.strftime('%m/%d/%Y')+'\n')

			driver.find_element_by_class_name('endDate').send_keys('\n')
			time.sleep(25)

			activities = driver.find_elements_by_class_name("activity")
			for j in range(int(driver.find_element_by_class_name('totalHits').text.strip(' results'))):
				
				activities[j].click()
				time.sleep(1)

				if driver.find_element_by_id('quickViewFileUploadDiv').text != 'Upload':
					driver.find_element_by_id('quickViewFileUploadDiv').click()
					driver.set_page_load_timeout(10)

					# download
					driver.find_element_by_class_name('download').click()
					driver.set_page_load_timeout(10)

				driver.find_element_by_id('closeIcon').click()
				driver.set_page_load_timeout(10)

		driver.find_element_by_class_name('closeIcon').click()
		driver.set_page_load_timeout(10)

		driver.find_element_by_class_name('groupAndAthleteSelector').click() # dropdown menu
		time.sleep(10)
		athletes = driver.find_elements_by_class_name('athleteOption') # list of athletes
	else:
		print("exclude")
		continue