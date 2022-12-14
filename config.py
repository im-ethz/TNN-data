import yaml

with open('credentials.yml', 'r') as f:
	credentials = yaml.safe_load(f) 

with open('mapping.yml', 'r') as f:
	rider_mapping = yaml.safe_load(f)

rider_mapping_inv = {v:k for k,v in rider_mapping.items()}

DATA_PATH = '/wave/hypex/data/'