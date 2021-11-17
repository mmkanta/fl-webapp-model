from model.classification_pylon_1024.predict_1024 import *

async def main(file_location, content_type):
    image_load = ""
    if 'dicom' in str(content_type) or 'octet-stream' in str(content_type):
        dicom, data = dicom2array(file_location)
        image_load = data
    elif 'png' in str(content_type):
        image_load = PIL.Image.open(file_location)
    
    path = os.path.join(os.path.dirname(file_location), 'result')
    if not os.path.exists(path):
        os.makedirs(path)

    cv2.imwrite(os.path.join(path, 'covid_image.png'), image_load)

    return [{"Admission": 0.08201, "Severity":0.001234}]