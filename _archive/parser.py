# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 14:55:40 2020

@author: Eva van Weenen (evanweenen@ethz.ch)
"""
import os
import pandas as pd
import numpy as np
import re

import fitparse
import xml.etree.ElementTree as ET

# OUTCOMMENT THESE LINES AND DEFINE MAPPING YOURSELF IF YOU ARE RUNNING THIS CODE OUTSIDE OF THE HYPEX PROJECT
from config import rider_mapping 
rider_mapping_inv = {v:k for k,v in rider_mapping.items()}

class XML(object):
    def __init__(self, filename, path_in, path_out):
        self.filename = filename
        self.path_in = path_in
        self.path_out = path_out

    def read_data(self, level=0):
        file = ET.parse(os.path.join(self.path_in, self.filename))
        root = file.getroot()
        for l in range(level):
            root = root[0]
        self.data = root
        self.space = re.match(r'{.*}', root.tag).group(0)
        self.name = root.tag.replace(self.space, '')
        self.categories = list(np.unique([cat.tag.replace(self.space, '') for cat in self.data]))

    def get_elements(self, cat):
        elements = self.data.findall(self.space+cat)
        self.categories.remove(cat)
        return elements

    def get_category(self, cat):
        df = pd.DataFrame()
        for i, element in enumerate(self.get_elements(cat)):
            for item in element:
                df.loc[i, item.tag.replace(self.space,'')] = item.text
        return df

    def save_file(self, df, cat):
        if not os.path.exists(os.path.join(self.path_out, cat)):
            os.makedirs(os.path.join(self.path_out, cat))
        df.to_csv(os.path.join(self.path_out, cat, os.path.splitext(self.filename)[0]+'_'+cat+'.csv'))

class PWX(XML):
    def __init__(self, filename, path_in, path_out):
        super().__init__(filename, path_in, path_out)

        self.fitnames = {   'sample'                :'record',
                            'event'                 :'event',
                            'segment'               :'lap',
                            'Bike'                  :'cycling',
                            'make'                  :'manufacturer',
                            'model'                 :'product',
                            'duration'              :'total_elapsed_time',
                            'distance'              :'total_distance',
                            'climbingelevation'     :'total_ascent',
                            'descendingelevation'   :'total_descent',
                            'hr'                    :'heart_rate',
                            'spd'                   :'speed',
                            'pwr'                   :'power',
                            'dist'                  :'distance',
                            'lat'                   :'position_lat',
                            'lon'                   :'position_long',
                            'alt'                   :'altitude',
                            'temp'                  :'temperature',
                            'cad'                   :'cadence',
                            'type'                  :'event_type'}

    def get_header(self):
        df = pd.Series(4*[{}], index=['file_id', 'device_0', 'session', 'sport'])

        # summary data
        if 'summarydata' in self.categories:
            summary = self.get_category('summarydata')
            if not summary.empty:
                df.loc['session'].update(summary.loc[0].to_dict())

        # sport type
        if 'sportType' in self.categories:
            item = self.get_elements('sportType')[0]
            df.loc['session'].update({'sport' : item.text})
            df.loc['sport'].update({'sport' : item.text})

        # device
        if 'device' in self.categories:
            device = self.get_category('device').loc[0].to_dict()
            df.loc['device_0'].update(device)
            df.loc['file_id'].update(device)
        
        # cmt
        if 'cmt' in self.categories:
            item = self.get_elements('cmt')[0]
            df.loc['device_0'].update({'descriptor' : item.text})

        # time
        if 'time' in self.categories:
            item = self.get_elements('time')[0]
            self.start_time = pd.to_datetime(item.text)
            df.loc['session'].update({'start_time' : self.start_time})
            df.loc['file_id'].update({'time_created' : self.start_time})

        # athlete
        if 'athlete' in self.categories:
            item = self.get_elements('athlete')[0][0]
            df.loc['file_id'].update({'athlete' : item.text})

        df = df.apply(pd.Series).stack().rename(index=self.fitnames).replace(self.fitnames)

        if 'beginning' in df['session']:
            df[('session', 'start_time')] += pd.to_timedelta(int(float(df[('session', 'beginning')])), unit='S')
            df = df.drop([('session', 'beginning')])

        if 'durationstopped' in df['session']:
            df[('session', 'total_timer_time')] = float(df[('session', 'total_elapsed_time')]) - float(df[('session', 'durationstopped')])
            df = df.drop([('session', 'durationstopped')])

        self.save_file(df, 'info')
        return df

    def get_laps(self):
        if 'segment' in self.categories:
            df = pd.DataFrame()
            for i, element in enumerate(self.get_elements('segment')): # each segment
                for item in element[1]:
                    df.loc[element[0].text, item.tag.replace(self.space,'')] = item.text
            df = df.rename(columns=self.fitnames)
            
            if 'beginning' in df and 'time' in self.categories:
                df['beginning'] = self.start_time + pd.to_timedelta(df['beginning'].astype(float).astype(int), unit='S')
                df = df.rename(columns={'beginning':'timestamp'})

            if 'durationstopped' in df and 'total_elapsed_time' in df:
                df['total_timer_time'] = df['total_elapsed_time'].astype(float) - df['durationstopped'].astype(float)
                df = df.drop('durationstopped', axis=1)

            self.save_file(df, 'lap')
            return df

    def get_events(self):
        if 'event' in self.categories:
            df = self.get_category('event')
            if 'timeoffset' in df:
                df['timestamp'] = self.start_time + pd.to_timedelta(df['timeoffset'].astype(float), unit='S')
                df = df.drop('timeoffset', axis=1)
                df = df.set_index('timestamp')
            df = df.rename(columns=self.fitnames)
            self.save_file(df, 'event')
            return df

    def get_records(self):
        if 'sample' in self.categories:
            df = self.get_category('sample')
            if 'timeoffset' in df:
                df['timestamp'] = self.start_time + pd.to_timedelta(df['timeoffset'].astype(float), unit='S')
                df = df.drop('timeoffset', axis=1)
                df = df.set_index('timestamp')
            df = df.rename(columns=self.fitnames)

            # convert degrees to semicircles (similar to .fit files)
            if 'position_lat' in df:
                df['position_lat'] = df['position_lat'].astype(float) / (180 / 2**31)
            if 'position_long' in df:
                df['position_long'] = df['position_long'].astype(float) / (180 / 2**31)

            self.save_file(df, 'record')
            return df

    def parse(self):
        self.read_data(level=1)
        self.get_header()
        self.get_laps()
        self.get_events()
        self.get_records()

        if len(self.categories) != 0:
            print("Message types not processed: ", *tuple(self.categories))

class TCX(XML):
    def __init__(self, filename, path_in, path_out):
        super().__init__(filename, path_in, path_out)

        self.fitnames = {'Name'                     : 'product_name',
                         'UnitId'                   : 'serial_number',
                         'ProductID'                : 'product',
                         'Time'                     : 'timestamp',
                         'PositionLatitudeDegrees'  : 'position_lat',
                         'PositionLongitudeDegrees' : 'position_long',
                         'AltitudeMeters'           : 'altitude',
                         'DistanceMeters'           : 'distance',
                         'Cadence'                  : 'cadence',
                         'HeartRateBpm'             : 'heart_rate',
                         'Speed'                    : 'speed',
                         'Watts'                    : 'power',
                         'TotalTimeSeconds'         : 'total_timer_time',
                         'DistanceMeters'           : 'total_distance',
                         'MaximumSpeed'             : 'max_speed',
                         'Calories'                 : 'total_calories',
                         'Intensity'                : 'intensity_factor',
                         'TriggerMethod'            : 'trigger',
                         'MaxBikeCadence'           : 'max_cadence',
                         'AvgSpeed'                 : 'avg_speed',
                         'AvgWatts'                 : 'avg_power',
                         'AverageHeartRateBpm'      : 'avg_heart_rate',
                         'MaxWatts'                 : 'max_power',
                         'MaximumHeartRateBpm'      : 'max_heart_rate'}

    def get_version(self, item):
        version = ''
        for subversion in ('VersionMajor', 'VersionMinor', 'BuildMajor', 'BuildMinor'):
            version += item.findall(self.space+subversion)[0].text + '.'
        return version.rstrip('.')

    def get_extension(self, item):
        df = pd.Series()
        for ext in item:
            space = re.match(r'{.*}', ext.tag).group(0)
            for col in ext:
                df[col.tag.replace(space, '')] = col.text
        return df

    def get_header(self):
        df = pd.Series(3*[{}], index=['file_id', 'device_0', 'session'])

        creator = {}
        # creator
        if 'Creator' in self.categories:
            creator['manufacturer'] = 'Garmin'
            elements = self.get_elements('Creator')[0]
            for item in elements:
                tag = item.tag.replace(self.space, '')
                if tag == 'Version':
                    creator['software_version'] = self.get_version(item)
                else:
                    creator[tag] = item.text
        # extensions
        if 'Extensions' in self.categories:
            extensions = self.get_elements('Extensions')[0]
            if len(extensions) > 0:
                for item in extensions:
                    creator[item.tag] = item.text
        df.loc['file_id'].update(creator)
        df.loc['device_0'].update(creator)

        # time
        if 'Id' in self.categories:
            item = self.get_elements('Id')[0]
            df.loc['file_id'].update({'time_created' : item.text})
            df.loc['session'].update({'start_time' : item.text})
        
        # training
        if 'Training' in self.categories:
            summary = {}
            elements = self.get_elements('Training')[0]
            for element in elements:
                for item in element:
                    if item.text is not None:
                        summary[item.tag.replace(self.space, '')] = item.text
            df.loc['session'].update(summary)

        df = df.apply(pd.Series).stack().rename(index=self.fitnames)
        self.save_file(df, 'info')
        return df

    def get_laps(self):
        df = pd.DataFrame()
        for i, element in enumerate(self.data.findall(self.space+'Lap')):
            for item in element:
                tag = item.tag.replace(self.space,'')
                if tag == 'Extensions':
                    for col, value in self.get_extension(item).iteritems():
                        df.loc[i, col] = value
                elif tag != 'Track':
                    df.loc[i, tag] = item.text
  
        df = df.rename(columns=self.fitnames)
        self.categories.remove('Lap')
        self.save_file(df, 'lap')
        return df

    def get_records(self):
        df = pd.DataFrame()

        c = 0
        for lap in self.data.findall(self.space+'Lap'):
            records = lap.findall(self.space+'Track')[0]

            for i, element in enumerate(records):
                for item in element:
                    tag = item.tag.replace(self.space,'')
                    if tag == 'Position':
                        for pos in item:
                            df.loc[c+i, tag+pos.tag.replace(self.space, '')] = pos.text                        
                    elif tag == 'Extensions':
                        for col, value in self.get_extension(item).iteritems():
                            df.loc[c+i, col] = value                  
                    else:
                        df.loc[c+i, tag] = item.text
            c += i
        
        df = df.rename(columns=self.fitnames)

        # convert degrees to semicircles (similar to .fit files)
        if 'position_lat' in df:
            df['position_lat'] = df['position_lat'].astype(float) / (180 / 2**31)
        if 'position_long' in df:
            df['position_long'] = df['position_long'].astype(float) / (180 / 2**31)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        self.save_file(df, 'record')
        return df

    def parse(self):
        self.read_data(level=2)
        self.get_header()
        self.get_laps()
        self.get_records()

        if len(self.categories) != 0:
            print("Message types not processed: ", *tuple(self.categories))

class FIT(object):
    def __init__(self, filename, path_in, path_out):
        self.filename = filename
        self.path_in = path_in
        self.path_out = path_out

    def read_data(self):
        file = fitparse.FitFile(os.path.join(self.path_in, self.filename))

        categories = []
        for message in file.messages:
            try:
                cat = message.mesg_type.name
            except:
                cat = message.mesg_type
            categories.append(cat)
        categories = pd.DataFrame(categories)[0]

        data_nans = np.array(file.messages)[categories.isna()] # extract nan messages

        self.categories = categories.unique().tolist()

        self.data = {cat: list(file.get_messages(cat)) for cat in self.categories}
        self.data.update({None:data_nans})

    def get_messages(self, df, cat):
        if cat in self.categories:
            for message in self.data[cat]:
                for field in message:
                    df.loc[cat, field.name] = field.value
            self.categories.remove(cat)
        return df

    def get_category(self, cat):
        if cat in self.categories:
            df = pd.DataFrame()
            for i, message in enumerate(self.data[cat]):
                for field in message.fields:
                    try:
                        df.loc[i,field.name] = field.value
                    except ValueError:
                        continue
            self.categories.remove(cat)
            self.save_file(df, cat)
            return df

    def get_hrv(self):
        if 'hrv' in self.categories:
            df = pd.DataFrame(columns=range(5))
            for i, message in enumerate(self.data['hrv']):
                for field in message.fields:
                    try:
                        df.loc[i] = field.value
                    except ValueError:
                        continue
            self.categories.remove('hrv')
            self.save_file(df, 'hrv')
            return df

    def get_laps(self):
        if 'lap' in self.categories:
            df = pd.DataFrame()
            for i, message in enumerate(self.data['lap']):
                for field in message.fields:
                    if type(field.value) != tuple:
                        df.loc[i,field.name] = field.value
                    else:
                        try:
                            df.at[i,field.name] = field.value
                            break
                        except:
                            df[field.name] = df[field.name].astype(object)
                            df.at[i,field.name] = field.value
                            break
                        else:
                            break
            self.categories.remove('lap')

            self.save_file(df, 'lap')

            return df

    def get_header(self, device):
        df = pd.Series(dtype=object)
        df.index = pd.MultiIndex.from_product([[], df.index])

        df = self.get_messages(df, 'file_id')
        df = self.get_messages(df, 'workout')
        df = self.get_messages(df, 'sport')
        df = self.get_messages(df, 'activity')
        df = self.get_messages(df, 'session')
        df = self.get_messages(df, 'training_file')
        df = self.get_messages(df, 'user_profile')
        df = self.get_messages(df, 'device_settings')
        df = self.get_messages(df, 'zones_target')
        df = self.get_messages(df, 'bike_profile')
        df = self.get_messages(df, 'weight_scale')

        # field description
        if 'field_description' in self.data:
            for message in self.data['field_description']:
                try:
                    df.loc['units', message.fields[3].value] = message.fields[4].value
                except IndexError:
                    continue
            self.categories.remove('field_description')

        # hr_zone
        if 'hr_zone' in self.data:
            for i, field in enumerate(self.data['hr_zone'][0].fields):
                if field.name == 'high_bpm':
                    hr_zone_field = i
            df.loc['hr_zone', self.data['hr_zone'][0].name+' [%s]'%self.data['hr_zone'][0].fields[hr_zone_field].units] = [message.fields[hr_zone_field].value for message in self.data['hr_zone']]
            self.categories.remove('hr_zone')

        # power_zone
        if 'power_zone' in self.data:
            for i, field in enumerate(self.data['power_zone'][0].fields):
                if field.name == 'high_value':
                    power_zone_field = i
            df.loc['power_zone', self.data['power_zone'][0].name+' [%s]'%self.data['power_zone'][0].fields[power_zone_field].units] = [message.fields[power_zone_field].value for message in self.data['power_zone']]
            self.categories.remove('power_zone')

        # device info
        if device is not None:
            if "serial_number" in device.columns:
                for i, item in enumerate(device.serial_number.dropna().unique()):
                    row = device[device.serial_number == item].dropna(axis=1).drop('timestamp', axis=1).drop_duplicates().iloc[0]
                    row.index = pd.MultiIndex.from_product([["device_%i"%i], row.index])
                    df = df.append(row)
            else:
                for i, item in device.iterrows():
                    item.index = pd.MultiIndex.from_product([["device_0"], item.index])
                    df = df.append(item)

        self.save_file(df, 'info')

        return df

    def save_file(self, df, cat):
        if not os.path.exists(os.path.join(self.path_out, str(cat))):
            os.makedirs(os.path.join(self.path_out, str(cat)))
        df.to_csv(os.path.join(self.path_out, str(cat), os.path.splitext(self.filename)[0]+'_'+str(cat)+'.csv'))

    def parse(self):
        try:
            self.read_data()
        except Exception as exception:
            print(type(exception).__name__, exception)
            assert type(exception).__name__ == 'FitEOFError' or type(exception).__name__ == 'FitCRCError'
            return

        device = self.get_category('device_info') # previously: device
        self.get_header(device)

        self.get_category('record') # previously: data
        self.get_category('hrv')
        self.get_category(None) # previously: nan
        self.get_category('event') # previously: startstop
        self.get_laps() # previosly: laps
       
        if len(self.categories) != 0:
            print("Message types not processed: ", *tuple(self.categories))

class Parser(object):
    """
    Parse file from any extension to csv
    Note that we need this class for joblib parallel 
    """
    def __init__(self, root, i):
        self._path_raw = f'{root}/export/{rider_mapping_inv[i]}/'
        self._path_fit = f'{root}/fit/{i}/'
        self._path_csv = f'{root}/csv/{i}/'

        if not os.path.exists(self._path_fit):
            os.makedirs(self._path_fit)
        if not os.path.exists(self._path_csv):
            os.makedirs(self._path_csv)

        self.FILE = {   '.fit'    : FIT,
                        '.FIT'    : FIT,
                        '.pwx'    : PWX,
                        '.tcx'    : TCX}

    def __parse_item__(self, file):
        # split xxx.fit | .gz
        filename, _ = os.path.splitext(file)
        # split xxx | .fit
        basename, extension = os.path.splitext(filename)

        if not os.path.exists(f'{self._path_csv}record/{basename}_record.csv'):
            print(filename)
            # unzip
            os.system(f"gzip -dk '{self._path_raw}/{file}'")
            # move
            os.system(f"mv '{self._path_raw}/{filename}' '{self._path_fit}/{filename}'")
            # parse to csv
            if extension in self.FILE and os.path.exists(os.path.join(self._path_fit, filename)):
                self.FILE[extension](filename, self._path_fit, self._path_csv).parse()
            else:
                return

def main():
    """
    Parse any file (.fit, .FIT, .pwx or .tcx) to a csv file.

    This code unzips .gz files, then converts this (.fit, .FIT, .pwx or .tcx) file to a csv file.
    So far, it is only used on Linux systems. If you are running the code on a different OS, and are running into problems,
    please contact the author.

    The code is designed so that all fit, pwx, and tcx files have roughly the same output csv files (e.g. same column names). 
    This makes it easier for analyzing them later.

    It assumes the following file structure:
    - {root}/export/{name_of_athlete_i} is where your .fit.gz files should be saved
    - {root}/fit/{i} is where the script will save your .fit files
    - {root}/csv/{message_type}/{i} is where the csv files will be saved

    {root} is the "root" directory under which you have saved your files, respectively to where you run the script from (!).
    
    Here, {name_of_athlete_i} is the name of the athlete, and {i} is the number that you give to the athlete to anonymize the files.
    The mapping from athlete name to number should be defined in the variable rider_mapping 
    (which for the case of the HYPEX project is imported from a python file "config").
    An example of rider_mapping is:
    rider_mapping = {'John Doe'             : 1,
                     'Albert Einstein'      : 2,
                     'Winston Churchill'    : 3}
    Of course, the names in the rider mapping should correspond exactly to your file structure in {root}/export/.
    """
    root = 'data/TrainingPeaks/' # TODO: adjust this to your root directory

    parser = Parser(root, i)

    for file in os.listdir(parser._path_raw):
        parser.__parse_item__(file)

if __name__ == '__main__':
    main()