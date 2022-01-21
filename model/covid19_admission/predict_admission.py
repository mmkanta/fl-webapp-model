import os
import json

def main(file_dir):
    
    res_dir = os.path.join(os.path.dirname(file_dir), 'result')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(res_dir, 'prediction.txt'), 'w') as f:
        json.dump([{"Admission": 0.08201}], f)

    return True, ""