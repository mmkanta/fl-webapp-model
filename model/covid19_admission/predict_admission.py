from model.classification_pylon_1024.predict_1024 import *

async def main(file_location, content_type):
    
    path = os.path.join(os.path.dirname(file_location), 'result')
    if not os.path.exists(path):
        os.makedirs(path)

    return [{"Admission": 0.08201}]