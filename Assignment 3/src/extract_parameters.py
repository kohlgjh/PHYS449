'''Function to get params from json file'''
import os
import json

def get_parameters(args):
    '''Extracts parameters from json file passed in argparse.

    Returns
    ---
    json_data: dict of json params
    '''
    json_path = os.path.join(os.getcwd(), str(args.param))
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        return json_data