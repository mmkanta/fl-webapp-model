import pandas as pd
# from glob import glob
import pydicom, numpy as np
import os, sys

# import matplotlib.pylab as plt
# from tqdm.notebook import tqdm
# from time import time
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from imageio import imwrite
from pathlib import Path

from datetime import datetime

# from Constant import AI_VERSION
AI_VERSION = "UTC_MDCU_Rad_v1.0.2.6"

def extract_dcm_info(ds):
    """
    Source: https://www.kaggle.com/thomasseleck/fastest-way-to-load-data-from-dicom-files
    This method extracts the content of a DCM file (image and metadata).

    Parameters
    ----------
    ds: pydicom.dataset.FileDataset
            object of DCM file

    Returns
    -------
    img: numpy array
            Extracted image from the DCM file.

    metadata_df: Pandas DataFrame
            DataFrame containing the extracted metadata.
    """


    # Extract the metadata
    metadata_dict = {}
    metadata_dict["Accession Number"] = ds.AccessionNumber

    # Extract meta data in file_meta first
    for elem in ds.file_meta:
        key_name = elem.name
        if elem.VR != 'SQ':
            if key_name not in metadata_dict:
                metadata_dict[key_name] = elem.value
                # Also store represent value if it exists
                if str(elem.value) != str(elem.repval.strip("'")):
                    metadata_dict[key_name+'_'+'repval'] = elem.repval.strip("'")
            else:
                metadata_dict[key_name+'_'+str(elem.tag)] = elem.value
        else:
            metadata_dict[key_name] = [dictify(item) for item in elem]
            
    # try:
    #     metadata_dict["Study Date"] = pd.to_datetime(ds.StudyDate+' '+ds.StudyTime.replace('.0',''), infer_datetime_format=True)
    # except:
    #     metadata_dict["Study Date"] = ds.StudyDate+' '+ds.StudyTime.replace('.0','')
    
    metadata_dict["Study Date"] = ds.StudyDate
    metadata_dict["Study Time"] = ds.StudyTime
        
    # metadata_dict["Patient"] = ds.PatientName.family_name + " " + ds.PatientName.given_name
    
    
    try:
        metadata_dict["DOB"] = pd.to_datetime(ds.PatientBirthDate, infer_datetime_format=True)
    except:
        metadata_dict["DOB"] = ds.PatientBirthDate
    
    if 'PatientAge' in ds:
        metadata_dict["Age"] = ds.PatientAge
    else:
        metadata_dict["Age"] = np.nan
        
    metadata_dict["Gender"] = ds.PatientSex
    metadata_dict["modality"] = ds.Modality
    
    if 'BodyPartExamined' in ds:
        metadata_dict["body_part_examined"] = ds.BodyPartExamined
    else:
        metadata_dict["body_part_examined"] = np.nan

    if 'ViewPosition' in ds:
        metadata_dict["view_position"] = ds.ViewPosition
    else:
        metadata_dict["view_position"] = np.nan

    try:
        metadata_dict['StudyID'] = ds[0x020,0x0010].value
    except:
        metadata_dict["StudyID"] = np.nan 
    
    if 'StudyDescription' in ds:
        metadata_dict["Procedure"] = ds.StudyDescription
    elif 'ProcedureCodeSequence' in ds:
        metadata_dict["Procedure"] = ds.ProcedureCodeSequence[0][('0008', '0104')].value
    elif 'PerformedProtocolCodeSequence' in ds:
        metadata_dict["Procedure"] = ds.PerformedProtocolCodeSequence[0][('0008', '0104')].value
    elif 'RequestAttributesSequence' in ds:
        metadata_dict["Procedure"] = ds.RequestAttributesSequence[0][('0040', '0007')].value #ds.ViewPosition
    
    if 'PhotometricInterpretation' in ds:
        metadata_dict["photometric"] = ds.PhotometricInterpretation
    else:
        metadata_dict["photometric"] = np.nan

    if "PixelData" in ds:
        rows = int(ds.Rows)
        cols = int(ds.Columns)
        metadata_dict["image_height"] = rows
        metadata_dict["image_width"] = cols
        metadata_dict["image_size"] = len(ds.PixelData)
    else:
        metadata_dict["image_height"] = np.nan
        metadata_dict["image_width"] = np.nan
        metadata_dict["image_size"] = np.nan
        
    if "ImagerPixelSpacing" in ds:
        metadata_dict["pixel_spacing_x"] = ds.ImagerPixelSpacing[0]
        metadata_dict["pixel_spacing_y"] = ds.ImagerPixelSpacing[1]
    else:
        metadata_dict["pixel_spacing_x"] = np.nan
        metadata_dict["pixel_spacing_y"] = np.nan
    
    if 'WindowCenter' in ds:
        metadata_dict["WindowCenter"] = ds.WindowCenter
        metadata_dict["WindowWidth"] = ds.WindowWidth
    else:
        metadata_dict["WindowCenter"] = np.nan
        metadata_dict["WindowWidth"] = np.nan
    
    if 'WindowCenterWidthExplanation' in ds:
        metadata_dict["WindowCenterWidthExplanation"] = ds.WindowCenterWidthExplanation
    else:
        metadata_dict["WindowCenterWidthExplanation"] = np.nan

    try:
        metadata_dict['StudyID'] = ds[0x020,0x0010].value
    except:
        metadata_dict["StudyID"] = np.nan

    metadata_dict["ID"] = ds.PatientID

    metadata_df = pd.DataFrame.from_records([metadata_dict])

    # Extract the image (in OpenCV BGR format)
#     img = cv2.cvtColor(ds.pixel_array, cv2.COLOR_GRAY2BGR)

    return metadata_df

# https://www.kaggle.com/sarmat/extract-metadata-from-dicom
def dictify(ds):
    output = dict()
    for elem in ds:
        if elem.VR != 'SQ': 
            output[elem.tag] = elem.value
        else:
            output[elem.tag] = [dictify(item) for item in elem]
    output_df = pd.DataFrame.from_records([output])
    return output, output_df

def create_path_and_save_dcm(ds, metadata_df):
    """
    Parameters
    ----------
    ds: pydicom.dataset.FileDataset, pydicom.dataset.Dataset
            object of DCM file
    metadata_df: Pandas DataFrame
            DataFrame containing the extracted metadata.

    Returns
    -------
    None
    
    """
    date_time = metadata_df['Study Date'][0]
    year = date_time.year
    month = date_time.month
#     day = date_time.day
    Accession_Number = ds.AccessionNumber

    folder_path = f'DICOMOBJ/{year}/{month}'
    
    Path(folder_path).mkdir(parents=True, exist_ok=True)

#     print(access_num, year, month, day)
    path_store_dicom = f'{folder_path}/{Accession_Number}.dcm'
    metadata_df['file_path'] = path_store_dicom
    i = 0
    while os.path.exists(path_store_dicom):
        i += 1
        file_name = path_store_dicom.split('_')[0]
        path_store_dicom = f'{file_name.strip(".dcm")}_({i}).dcm'
        metadata_df.loc[0, 'Accession Number'] = f'{Accession_Number}_({i})'
        metadata_df.loc[0, 'file_path'] = f'{folder_path}/{Accession_Number}_({i}).dcm'
    print('Save DICOM file at:', path_store_dicom)
    ds.save_as(path_store_dicom, write_like_original=False)
    
    path_store_meta_data = 'database/meta_data.csv'
    Path('database').mkdir(parents=True, exist_ok=True)
    print(f'Save meta_data file at: {path_store_meta_data}')
    if not os.path.exists(path_store_meta_data):
        metadata_df.to_csv(path_store_meta_data, index=False)
    else:
        metadata_df.to_csv(path_store_meta_data, index=False, mode='a', header=False)
    return
    
def create_path_and_save_png(ds, metadata_df, SIZE = 1024):
    """
    Parameters
    ----------
    ds: pydicom.dataset.FileDataset, pydicom.dataset.Dataset
            object of DCM file
    metadata_df: Pandas DataFrame
            DataFrame containing the extracted metadata.

    Returns
    -------
    None
    
    """
    # date_time = metadata_df['Study Date'][0]
    # datetime object containing current date and time
    # date_time = datetime.now()
    # year = date_time.year
    # month = date_time.month
#     day = date_time.day
    Accession_Number = ds.AccessionNumber

    # folder_path = f'PNGOBJ/{year}/{month}'
    
    # Path(folder_path).mkdir(parents=True, exist_ok=True)

#     print(access_num, year, month, day)
    # path_store_png = f'{folder_path}/{Accession_Number}.png'
    # metadata_df.loc[0, 'file_path'] = path_store_png
    # i = 0
    # while os.path.exists(path_store_png):
    #     i += 1
    #     file_name = path_store_png.split('_')[0]
    #     path_store_png = f'{file_name.strip(".png")}_({i}).png'
    #     metadata_df.loc[0, 'Accession Number'] = f'{Accession_Number}_({i})'
    #     metadata_df.loc[0, 'file_path'] = path_store_png
        
    # print('Save PNG file at:', path_store_png)
    # ds, image = dicom2array(ds)
    # image_r = resize_image(image, size=SIZE, keep_ratio=True)
    # imwrite(path_store_png, image_r)
    
    path_store_meta_data = 'database/meta_data.csv'
    Path('database').mkdir(parents=True, exist_ok=True)
    print(f'Save meta_data file at: {path_store_meta_data}')
    if not os.path.exists(path_store_meta_data):
        metadata_df.to_csv(path_store_meta_data, index=False)
    else:
        metadata_df.to_csv(path_store_meta_data, index=False, mode='a', header=False)
    return
    
def modify_dicom(ds, modifyDicomNPArray):

    # ยังมีปัญหาตอนเอา array เข้า DICOM ผ่าน pydicom ที่ทำให้สีเพี้ยน และ PACS อ่านไม่ได้

    import pydicom
    import PIL
    import io
    from pydicom.uid import generate_uid
#     from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pynetdicom.sop_class import DigitalXRayImagePresentationStorage, SecondaryCaptureImageStorage
    from pydicom.encaps import encapsulate

    new_SOP_UID = generate_uid()

#     print("Setting file meta information...")
    # Modify required values for file meta information
    ds.file_meta.MediaStorageSOPInstanceUID = new_SOP_UID
    # file_meta.ImplementationClassUID = "1.2.3.4"
    ds.file_meta.ImplementationVersionName = AI_VERSION

    ds.SOPInstanceUID = new_SOP_UID

    # Set the transfer syntax
    # ds.is_little_endian = True
    # ds.is_implicit_VR = True # If set this to True the transfer syntax will be set to "Implicit VR Little Endian" only

    # Get height and width
    im_frame = PIL.Image.fromarray(modifyDicomNPArray)
#     print("image mode:", im_frame.mode)
    ds.Rows = im_frame.height
    ds.Columns = im_frame.width

    ds.WindowCenter = 127.0
    ds.WindowWidth = 255.0

    # Create compressed image
    frame_data = []
    with io.BytesIO() as output:
        im_frame.save(output, format="JPEG")
        frame_data.append(output.getvalue())

    ds.PhotometricInterpretation = "YBR_FULL" # "YBR_FULL_422" #"YBR_PARTIAL_422" #"YBR_ICT" #  "YBR_RCT" # "RGB" #
    ds.SamplesPerPixel = 3
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit =  ds.BitsStored - 1
    ds.PixelRepresentation = 0
    ds.add_new((0x0028,0x0006),"US",0) # set PlanarConfiguration
    ds.PlanarConfiguration = 0 # may not be required?

    # Ensure the file meta info exists and has the correct values for transfer syntax and media storage UIDs.
    ds.fix_meta_info()

    ds.PixelData = encapsulate(frame_data)

    ds['PixelData'].is_undefined_length = True

    return ds


def array_to_dicom(ds, dir, filename):
    import pydicom
    import PIL
    import io
    import subprocess

    Accession_Number = ds.AccessionNumber

    file_in = os.path.join(dir, filename)
    file_out = os.path.join(dir, filename.split('.')[0] + '.dcm')
    
    # Convert PNG to DICOM with Transfer syntax (Compress to JPEG)
    command =  f"gdcm2vtk --modality DX --jpeg {file_in} {file_out}"
    subprocess.run(command.split())
    # os.system(command)
    
    # Read Compressed DICOM
    ds_modify = pydicom.dcmread(file_out)
    
    # Add necessary tag
    ds_modify.AccessionNumber = ds.AccessionNumber
    ds_modify.PatientID = ds.PatientID
    ds_modify.PatientName = ds.PatientName
    ds_modify.PatientBirthDate = ds.PatientBirthDate
    ds_modify.PatientSex = ds.PatientSex
    ds_modify.file_meta.ImplementationVersionName = AI_VERSION

    # Fix time to follow the original DICOM file (Due to time in DICOM Container not corespond to local time)
    ds_modify.StudyDate = ds.StudyDate
    ds_modify.ContentDate = ds.ContentDate
    ds_modify.StudyTime = ds.StudyTime
    ds_modify.ContentTime = ds.ContentTime
    
    # Save DICOM with heatmap to disk
    dcm_compressed_add = os.path.join(dir, 'compressed_' + filename.split('.')[0] + '.dcm')
    ds_modify.save_as(dcm_compressed_add, write_like_original=False)

    # Delete Previous Compressed DICOM
    os.remove(file_in)
    os.remove(file_out)

    print("Get heatmap array to dicom complete")
    print(ds_modify)
    
    return ds_modify, dcm_compressed_add

def check_dicom_exist(acc_no):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOCAL_DIR = os.path.join(BASE_DIR, 'resources', 'local')

    acc_no = str(acc_no)
    for file in os.listdir(LOCAL_DIR):
        if file.endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(LOCAL_DIR, file))
            if ds.AccessionNumber == acc_no:
                filename = os.fsdecode(file)
                return True, filename
    else:
        return False, ""

def plot_bbox_from_df(df_bbox, dicom, image_path):

    ### df_bbox คือ json ที่ได้มาจาก database
    ### image_path คือ path ของ image

    import cv2
    import pydicom
    import numpy as np
    import pandas as pd
    import random

    all_bbox = pd.DataFrame(df_bbox['data'], columns=['label', 'tool', 'data'])

    #generate random class-color mapping
    all_class = all_bbox["label"].unique()
    cmap = dict()
    for lb in all_class:
        while True:
            b = random.randint(0,255)
            g = random.randint(0,255)
            r = random.randint(0,255)
            if (b,g,r) not in cmap.values():
                cmap[lb] = (b,g,r)
                break
    all_bbox["color"] = all_bbox["label"].map(cmap)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5
    thickness = 6
    isClosed = True

    ### ส่วนที่เรียก image
    #import dicom image array
    if dicom != None:
        # dicom = pydicom.dcmread(("img_path"))
        inputImage = dicom.pixel_array

        # depending on this value, X-ray may look inverted - fix that:
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            inputImage = np.amax(inputImage) - inputImage

        inputImage = np.stack([inputImage, inputImage, inputImage])
        inputImage = inputImage.astype('float32')
        inputImage = inputImage - inputImage.min()
        inputImage = inputImage / inputImage.max()
        inputImage = inputImage.transpose(1, 2, 0)
        inputImage = (inputImage*255).astype(int)
        # https://github.com/opencv/opencv/issues/14866
        inputImage = cv2.UMat(inputImage).get()
    else:
        inputImage = np.zeros([3000, 3000, 3])
        inputImage = inputImage.astype(int)
        inputImage = cv2.UMat(inputImage).get()

    all_ratio = []
    for index, row in all_bbox.iterrows():

        if row['tool'] == 'rectangleRoi':
            pts = row["data"]["handles"]
            inputImage = cv2.rectangle(inputImage, (int(pts["start"]["x"]), int(pts["start"]["y"])), (int(
                pts["end"]["x"]), int(pts["end"]["y"])), row["color"], thickness)
            inputImage = cv2.putText(inputImage,  row["label"], (int(min(pts["start"]["x"], pts["end"]["x"])), int(
                min(pts["start"]["y"], pts["end"]["y"]))), font, fontScale, row["color"], thickness, cv2.LINE_AA)

        if row['tool'] == 'freehand':
            pts = np.array([[cdn["x"], cdn["y"]]
                           for cdn in row["data"]["handles"]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            #choose min x,y for text origin
            text_org = np.amin(pts, axis=0)
            inputImage = cv2.polylines(inputImage, [pts], isClosed, row["color"], thickness)
            inputImage = cv2.putText(inputImage,  row["label"], tuple(
                text_org[0]), font, fontScale, row["color"], thickness, cv2.LINE_AA)

        if row['tool'] == 'length':
            pts = row["data"]["handles"]
            #choose left point for text origin
            text_org = "start"
            if pts["start"]["x"] > pts["end"]["x"]:
                text_org = "end"
            inputImage = cv2.line(inputImage, (int(pts["start"]["x"]), int(pts["start"]["y"])), (int(
                pts["end"]["x"]), int(pts["end"]["y"])), row["color"], thickness)
            inputImage = cv2.putText(inputImage,  row["label"], (int(pts[text_org]["x"]), int(
                pts[text_org]["y"])), font, fontScale, row["color"], thickness, cv2.LINE_AA)

        if row['tool'] == 'ratio':
            all_ratio.append((row["label"],row["data"]["ratio"],row["color"]))

        if row['tool'] == 'arrowAnnotate':
            pts = row["data"]["handles"]
            text_org = "start"
            if pts["start"]["x"] > pts["end"]["x"]: text_org = "end"
            inputImage = cv2.arrowedLine(inputImage, (int(pts["end"]["x"]), int(pts["end"]["y"])),(int(pts["start"]["x"]), int(pts["start"]["y"])), row["color"],thickness)
            inputImage = cv2.putText(inputImage,  row["label"], (int(pts["end"]["x"]),int(pts["end"]["y"])), font, fontScale, row["color"], thickness, cv2.LINE_AA)
    
    #write all ratios on top-left
    for i in range(len(all_ratio)):
        inputImage = cv2.putText(inputImage,  "{} ({})".format(all_ratio[i][0],all_ratio[i][1]), (10,50+60*i), font, fontScale, all_ratio[i][2], thickness, cv2.LINE_AA)

    #cv2.imshow(inputImage)
    cv2.imwrite(image_path, inputImage)
    return True