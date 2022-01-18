from config import credentials, rider_mapping
import pyderman as dr
from selenium import webdriver
from tqdm import tqdm

import pandas as pd 
import time

class Scraper(object):
	def __init__(self, dates, month_first=True):
		self.dates = dates
		self.restart = True
		if month_first:
			self.date_format = '%m/%d/%Y'
		else:
			self.date_format = '%d/%m/%Y'

	def open_driver(self, download_path):
		driver_path = dr.install(browser=dr.chrome, file_directory='./lib/', verbose=True, chmod=True, 
			overwrite=False, version='96.0.4664.45', filename=None, return_info=False)
		options = webdriver.ChromeOptions()
		options.add_experimental_option("prefs", {"download.default_directory":download_path})
		self.driver = webdriver.Chrome(driver_path, options=options)
		self.driver.maximize_window()

	def login(self, url_login, credentials):
		self.driver.get(url_login)

		self.driver.find_element_by_name('Username').send_keys(credentials['username'])
		self.driver.find_element_by_name('Password').send_keys(credentials['password'])
		self.driver.find_element_by_name('submit').click()
		
		time.sleep(15)

	def click_calendar(self):
		self.driver.find_element_by_class_name('calendar').click()
		time.sleep(5)

	def get_athletes(self):
		# find athletes
		self.driver.find_element_by_class_name('groupAndAthleteSelector').click() # dropdown menu
		time.sleep(5)
		self.athletes = self.driver.find_elements_by_class_name('athleteOption') # list of athletes

	def check_ethics(self, i):
		print(self.athletes[i].text)
		if ' '.join(self.athletes[i].text.split()[1:]).lower() in rider_mapping.keys() or \
			self.athletes[i].text.lower() == 'kusztor peter':
			print("include")
			return True
		else:
			print("exclude")
			return False

	def click_athlete(self, i):
		self.athletes[i].click() # click on ith athlete
		time.sleep(5)

	def click_workouts(self):
		# go to list layout
		self.driver.find_element_by_class_name('workoutSearch').click()
		time.sleep(5)

		# select only bike training (only first time visiting website)
		if self.restart:
			self.driver.find_element_by_class_name('filter').click()
			self.driver.set_page_load_timeout(10)

			self.driver.find_element_by_xpath("//*[@id='main']/div[1]/div/div/div[3]/div[3]/div/div[2]/div[5]/div[4]/div[2]/label[2]").click() # select bike
			self.driver.set_page_load_timeout(10)

	def select_dates(self):
		start_date = self.dates[self.d]
		end_date = self.dates[self.d+1]

		self.driver.find_element_by_class_name('endDate').clear()
		self.driver.find_element_by_class_name('endDate').send_keys(end_date.strftime(self.date_format)+'\n')
		time.sleep(5)

		self.driver.find_element_by_class_name('startDate').clear()
		self.driver.find_element_by_class_name('startDate').send_keys(start_date.strftime(self.date_format)+'\n')

		self.driver.find_element_by_class_name('endDate').send_keys('\n')
		time.sleep(25)

	def scrape_activities(self, j_min=0):
		activities = self.driver.find_elements_by_class_name("activity")
		for self.j in range(j_min, int(self.driver.find_element_by_class_name('totalHits').text.strip(' results'))):
			
			activities[self.j].click()
			time.sleep(1)

			if self.driver.find_element_by_id('quickViewFileUploadDiv').text != 'Upload':
				self.driver.find_element_by_id('quickViewFileUploadDiv').click()
				self.driver.set_page_load_timeout(10)

				# download
				self.driver.find_element_by_class_name('download').click()
				self.driver.set_page_load_timeout(10)

			self.driver.find_element_by_id('closeIcon').click()
			self.driver.set_page_load_timeout(10)

	def scrape_athlete(self, d_min=0):
		# scrape trainingpeaks data for athlete i
		
		for self.d in tqdm(range(d_min, len(self.dates)-1)):
			self.select_dates()
			self.scrape_activities()

		self.driver.find_element_by_class_name('closeIcon').click()
		self.driver.set_page_load_timeout(10)

		self.restart = False

def main():
	"""
	Scrape Trainingpeaks
	Select by:
	- workout type = bike
	- date range = 01/01/2020 until 31/10/2020 -> 2019
	"""
	scraper = Scraper(dates=pd.date_range(start='01-01-2014', end='31-12-2021', freq='30D'))

	scraper.open_driver(download_path='/local/home/evanweenen/hype-data/data/TrainingPeaks/export/')
	scraper.login(url_login='https://home.trainingpeaks.com/login', credentials=credentials['TP']['pro'])

	scraper.click_calendar()

	scraper.get_athletes()

	for i in range(len(scraper.athletes)):
		if scraper.check_ethics(i)
			scraper.click_athlete(i)
			scraper.click_workouts()

			scraper.scrape_athlete()
			scraper.get_athletes()
		else:
			continue

if __name__ == '__main__':
    main()