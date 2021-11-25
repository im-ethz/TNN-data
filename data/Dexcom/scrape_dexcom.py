from config import credentials, rider_mapping
import pyderman as dr
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import pandas as pd
import datetime
import time

path = '/local/home/evanweenen/hype-data/data/Dexcom/'
dates = pd.date_range(start='01-01-2014', end='31-12-2021', freq='90D')

athletes = pd.read_csv(path+'dexcom_id.csv', index_col=0, dtype={'US':object, 'EU':object}) # beadle missing

# ------------------------ Dexcom CLARITY US
url_login = 'https://clarity.dexcom.com/professional/'

download_path = path+'export/US/'

driver_path = dr.install(browser=dr.chrome, file_directory='./lib/', verbose=True, chmod=True, overwrite=False, version='86.0.4240.22', filename=None, return_info=False)
options = webdriver.ChromeOptions()
options.add_experimental_option("prefs", {"download.default_directory":download_path})
driver = webdriver.Chrome(driver_path, options=options)
driver.maximize_window()

driver.get(url_login)

# login
driver.find_element_by_name('username').send_keys(credentials['dexcom']['US']['username'])
driver.find_element_by_name('password').send_keys(credentials['dexcom']['US']['password'])
driver.find_element_by_class_name('btn-sm').click()
time.sleep(5)

for d in range(len(dates)-1):

    start_date = dates[d]
    end_date = dates[d+1] - pd.to_timedelta('1D')

    for name, pid in athletes['US'].dropna().items():

        # this is alternative to typing in name in search bar
        driver.get(url_login+'patients/'+pid+'/export')
        time.sleep(20)

        # select date
        driver.find_element_by_css_selector('date-range-picker').click()
        driver.find_element_by_name('start_date').clear()
        driver.find_element_by_name('end_date').clear()
        driver.find_element_by_name('start_date').send_keys(start_date.strftime('%m/%d/%Y'))
        driver.find_element_by_name('end_date').send_keys(end_date.strftime('%m/%d/%Y'))

        driver.find_element_by_class_name('ok').click()

        # export file
        driver.find_element_by_name('submitExport').click()

driver.close()

# ------------------------ Dexcom CLARITY EU
url_login = 'https://clarity.dexcom.eu/professional/'

download_path = path+'export/EU/'

driver_path = dr.install(browser=dr.chrome, file_directory='./lib/', verbose=True, chmod=True, overwrite=False, version='86.0.4240.22', filename=None, return_info=False)
options = webdriver.ChromeOptions()
options.add_experimental_option("prefs", {"download.default_directory":download_path})
driver = webdriver.Chrome(driver_path, options=options)
driver.maximize_window()

driver.get(url_login)

# login
driver.find_element_by_name('username').send_keys(credentials['dexcom']['EU']['username'])
driver.find_element_by_name('password').send_keys(credentials['dexcom']['EU']['password'])
driver.find_element_by_class_name('btn-sm').click()
time.sleep(5)

for d in range(len(dates)-1):

    start_date = dates[d]
    end_date = dates[d+1] - pd.to_timedelta('1D')

    for name, pid in athletes['EU'].dropna().items():

        # this is alternative to typing in name in search bar
        driver.get(url_login+'patients/'+pid+'/export')
        time.sleep(20)

        # select date
        driver.find_element_by_css_selector('date-range-picker').click()
        driver.find_element_by_name('start_date').clear()
        driver.find_element_by_name('end_date').clear()
        driver.find_element_by_name('start_date').send_keys(start_date.strftime('%m/%d/%Y'))
        driver.find_element_by_name('end_date').send_keys(end_date.strftime('%m/%d/%Y'))

        driver.find_element_by_class_name('ok').click()

        # export file
        driver.find_element_by_name('submitExport').click()

driver.close()