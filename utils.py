import random
import numpy as np
import uuid
from datetime import datetime

def set_seed(seed: int):
    random.seed(seed)           
    np.random.seed(seed)       

def generate_random_folder_name():
    return str(uuid.uuid4())

def generate_timestamped_folder_name():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_finish_action(output):
    action = ''
    if output.startswith('Action:'):
        action = output.split("Action:")[1].split("\n")[0].strip()
    elif output.startswith('finish:'):
        ag_model = output.split('finish:')[1].strip()
        action = f'finish({ag_model})'
        output = f'Action: {action}'
    elif output.startswith('finish('):
        action = f'{output}'
        output = f'Action: {action}'
    else:
        action = 'finish(' + output.split('\n')[0]
        action = action.strip()
        output = 'Action: finish' + output
    return action, output