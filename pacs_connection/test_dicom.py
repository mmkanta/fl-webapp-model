import os
import pydicom
import pickle
import time
import pandas as pd
import datetime, time

from pynetdicom import AE, evt, StoragePresentationContexts, VerificationPresentationContexts, DEFAULT_TRANSFER_SYNTAXES, ALL_TRANSFER_SYNTAXES
# from pynetdicom.sop_class import DigitalXRayImagePresentationStorage, VerificationSOPClass, ComputedRadiographyImageStorage
from pynetdicom.pdu_primitives import SCP_SCU_RoleSelectionNegotiation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

i = 0
names = ['Linnet Campo', 'Marika McNiven', 'Thera Sharp']

# for file in os.listdir(os.path.join(BASE_DIR, 'resources', 'local')):
#     i = i + 1
#     filename = os.fsdecode(file)
#     print(filename)
#     ds = pydicom.dcmread(os.path.join(BASE_DIR, 'resources', 'local', filename))
#     print(ds.AccessionNumber)
#     print(ds.PatientID)
#     print(ds.PatientName)
#     print(ds.Modality)
#     print(ds.ProcedureCodeSequence)
#     print(pd.to_datetime(ds.StudyDate, infer_datetime_format=True))
#     print(ds.file_meta.MediaStorageSOPInstanceUID)

# ds = pydicom.dcmread(os.path.join(BASE_DIR, 'resources', 'local', '0043443.dcm'))
# ds.AccessionNumber = '0041018'
# ds.PatientID = '8789'
# ds.PatientName = 'Serina^Harford'
# ds.save_as(os.path.join(BASE_DIR, 'resources', 'local', '0041099.dcm'), write_like_original=False)

# ds = pydicom.dcmread(os.path.join(BASE_DIR, 'resources', 'local', '0041018.dcm'))
# print(ds.StudyDate)
# print(ds.PatientID)
# print(ds.PatientName)
# print(ds.Modality)
# print(ds.StudyDate)
# print(ds.ProcedureCodeSequence)

class FakeEvent:    
    def __init__(self, event):
        self.file_meta = event.file_meta
        self.dataset = event.dataset
        self.timestamp = event.timestamp
        self.assoc = FakeAssoc(event.assoc)


class FakeAssoc:    
    def __init__(self, assoc):
        self.requestor = FakeRequester(assoc.requestor)


class FakeRequester:    
    def __init__(self, requestor):
        self.ae_title = requestor.ae_title
        self.address = requestor.address
        self.port = requestor.port

def load_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_event(
                path_backup_event, 
                max_try=5,
                n_try=1,
                try_interval=2,
                ):
      
    event = load_file(path_backup_event)
    return event

def save_file(file, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(file, f)

# dicom = pydicom.read_file(os.path.join(BASE_DIR, 'resources', 'files', '0041018.dcm'))

# event = read_event(os.path.join(BASE_DIR, 'resources', 'files', '20211018CR0846.evt'))
# # event.file_meta = dicom.file_meta
# event.dataset.PatientID = '1234570'
# event.dataset.PatientSex = 'F'
# event.dataset.PatientAge = '30'
# event.dataset.PatientName = 'Serina^Harford'
# fake_event = FakeEvent(event)
# save_file(fake_event, os.path.join(BASE_DIR, 'resources', 'files', '20211018CR0846_2.evt'))

# for backup in os.listdir(os.path.join(BASE_DIR, 'resources', 'files')):
#     if backup.endswith('.evt'):
event = read_event(os.path.join(BASE_DIR, 'resources', 'files', '20211018CR0846.evt'))
ds = event.dataset
ds.file_meta = event.file_meta
if isinstance(ds, (pydicom.FileDataset, pydicom.dataset.Dataset)):
    print('is instance')
print(ds.AccessionNumber)
print(ds.StudyDate)
print(ds.StudyTime)
print(ds.PatientBirthDate)
print(ds.PatientAge)
print(ds.PatientSex)
print(ds.Modality)
print(ds.BodyPartExamined)
print(ds.ViewPosition)
print(ds.StudyDescription)
print(ds.ProcedureCodeSequence)
print(ds.PerformedProtocolCodeSequence)
# print(ds.PixelData)
print(ds.ImagerPixelSpacing)
print(ds.WindowCenter)
print(ds.WindowWidth)
# print(ds.WindowCenterWidthExplanation)
print(ds[0x020,0x0010].value)
print(ds.PatientID)
print(ds.PatientName)
print(ds.PatientName.family_name + " " + ds.PatientName.given_name)
print(ds.PatientSex)
print(ds.PatientAge)
print(ds.PatientBirthDate)
print(ds.StudyDate)
print(pd.to_datetime(ds.StudyDate, infer_datetime_format=True))
# print(pd.to_datetime(1642926950790))
print(datetime.datetime.fromtimestamp(float(ds.StudyTime)))
# ds.save_as(os.path.join(BASE_DIR, 'resources', 'files', 'save_test.dcm'))
print(event.assoc.requestor.ae_title)
# print(ds)