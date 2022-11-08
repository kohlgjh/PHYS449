'''Function to get params from json file'''
import os
import json

def get_parameters(args):
    '''Extracts hyperparameters from json file passed in argparse.

    Returns
    ---
    json_data: dict of json params
    '''
    json_path = os.path.join(os.getcwd(), str(args.json_path))
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        return json_data