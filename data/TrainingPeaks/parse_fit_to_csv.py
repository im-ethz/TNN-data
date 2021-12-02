"""
Parse Garmin FIT files to CSV
Makes use of the library https://github.com/dtcooper/python-fitparse
Uses the following message types:
- file_id
- workout
- sport
- activity
- session
- training_file
- user_profile
- device_settings
- zones_target
- bike_profile
- weight_scale
- field_description
- hr_zone
- power_zone
- record
- device_info
- event
- lap
- hrv

The following message types are ignored:
- file creator - doesn't contain anything interesting it seems
"""
import fitparse
import argparse
import numpy as np
import pandas as pd
import os

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
        df.to_csv(os.path.join(self.path_out, str(cat), self.filename.rstrip('.fit')+'_'+str(cat)+'.csv'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help="Name of the fitfile to convert")
    parser.add_argument('-i', '--input', type=str, default='', help="Path to the directory where the fitfile is located")
    parser.add_argument('-o', '--output', type=str, default='', help="Path to the directory where to save the csv")
    args = parser.parse_args()

    fit = FIT(args.filename, args.input, args.output)
    fit.read_data()

    device = fit.get_category('device_info') # previously: device
    fit.get_header(device)

    fit.get_category('record') # previously: data
    fit.get_category('hrv')
    fit.get_category(None) # previously: nan
    fit.get_category('event') # previously: startstop
    fit.get_laps() # previosly: laps
   
    if len(fit.categories) != 0:
        print("Message types not processed: ", *tuple(fit.categories))

if __name__ == '__main__':
    main()