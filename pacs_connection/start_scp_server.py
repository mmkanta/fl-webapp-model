from datetime import datetime
from Utilis_DICOM import *
from evt_classes import *
import subprocess
import traceback
import time
import pickle
import threading

import gc

from pydicom.uid import JPEGLosslessSV1
from pynetdicom import AE, evt, StoragePresentationContexts, VerificationPresentationContexts, DEFAULT_TRANSFER_SYNTAXES, ALL_TRANSFER_SYNTAXES
from pynetdicom.sop_class import ComputedRadiographyImageStorage, Verification, DigitalXRayImageStorageForPresentation
from pynetdicom.pdu_primitives import SCP_SCU_RoleSelectionNegotiation

import logging
from pynetdicom import debug_logger
# Setup logging to use the StreamHandler at the debug level
debug_logger()

import warnings
warnings.filterwarnings("ignore")

import argparse
# https://zetcode.com/python/argparse/
parser = argparse.ArgumentParser()
   
parser.add_argument('-s', '--host', type=str, default='', help="Host IP")
parser.add_argument('-p', '--port', type=int, default=11112, help="listening port number")

args = parser.parse_args()

ae = AE(ae_title=b'DEEPMEDWEBAPP')

SUPPORTED_ABSTRACT_SYNTAXES = [DigitalXRayImageStorageForPresentation, ComputedRadiographyImageStorage, Verification]
# SUPPORTED_ABSTRACT_SYNTAXES = [DigitalXRayImagePresentationStorage, ComputedRadiographyImageStorage, VerificationSOPClass]

ae.supported_contexts = StoragePresentationContexts # All abstract syntax except Verification SOP Class
ae.supported_contexts = VerificationPresentationContexts # Add Verification SOP Class ให้ SCU สามารถ echo มาได้

# Adding a presentation context with multiple transfer syntaxes
for abstract_syntax in SUPPORTED_ABSTRACT_SYNTAXES: # Adding additional transfer syntaxes by combine with the pre-existing context
    ae.add_supported_context(abstract_syntax, ALL_TRANSFER_SYNTAXES)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP_DIR = os.path.join(BASE_DIR, 'resources', 'backup')
            
def current_dt():
    return datetime.now().isoformat()


def save_file(file, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(file, f)


def load_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def backup_event(
                event,
                backup_dir, 
                max_try=5,
                n_try=1,
                try_interval=2,
                ):

    acc_no = event.dataset.AccessionNumber
    path_backup_event = os.path.join(backup_dir, f'{acc_no}.evt')
    fake_event = FakeEvent(event)

    while n_try <= max_try:

        try:
                        
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
            save_file(fake_event, path_backup_event)
            time.sleep(1)
            load_file(path_backup_event)
            print(f"{current_dt()} [Backup ds] {acc_no} Successfully backup")
            break

        except Exception as e:
            traceback.print_exc()
            print(f"{current_dt()} [Backup ds] {acc_no} Try #{n_try} Failed.\nSleep for 2 seconds and re-try again")

        n_try += 1
        time.sleep(try_interval)

    if n_try > max_try:
        print(f"{current_dt()} [Backup ds] {acc_no} Try #{max_try} Failed.\nSkip backing up this DICOM file")

    return path_backup_event


def clear_backup(
                file_path,
                max_try=5,
                n_try=1,
                try_interval=2,
                ):

    while n_try <= max_try:

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"{current_dt()} [Clear Backup] Successfully clear {file_path}")        

            break

        except Exception as e:
            traceback.print_exc()
            print(f"{current_dt()} [Clear Backup] {file_path} Try #{n_try} Failed.\nSleep for 2 seconds and re-try again")

        n_try += 1
        time.sleep(try_interval)

    if n_try > max_try:
        print(f"{current_dt()} [Clear Backup] {file_path} Try #{max_try} Failed.\nSkip clearing backup")

def read_event(
                path_backup_event, 
                max_try=5,
                n_try=1,
                try_interval=2,
                ):

    event = None

    while n_try <= max_try:

        try:
                        
            event = load_file(path_backup_event)
            print(f"{current_dt()} [Read backup ds] {path_backup_event} Successfully read")
            break

        except Exception as e:
            traceback.print_exc()
            print(f"{current_dt()} [Read backup ds] {path_backup_event} Try #{n_try} Failed.\nSleep for 2 seconds and re-try again")

        n_try += 1
        time.sleep(try_interval)

    if n_try > max_try:
        print(f"{current_dt()} [Read backup ds] {path_backup_event} Try #{max_try} Failed.\nSkip reading")

    return event


def startup_resend(backup_dir, resend_interval=30):
    for backup in os.listdir(backup_dir):
        if backup.endswith('.evt'):
            path_backup_event = os.path.join(backup_dir, backup)
            event = read_event(path_backup_event)
            if event is not None:
                handle_store(event)
                time.sleep(resend_interval)
    

def handle_store(event):
    # https://github.com/pydicom/pynetdicom/issues/487
    """Handle a C-STORE service request"""
    # SCP Role
    start_time = time.time()

    # Backup event    
    path_backup_event = backup_event(event, BACKUP_DIR)

    # Decode the C-STORE request's *Data Set* parameter to a pydicom Dataset
    ds = event.dataset
    # Add the File Meta Information
    ds.file_meta = event.file_meta

    Accession_Number = ds.AccessionNumber

    # Every *Event* includes `assoc` and `timestamp` attributes
    #   which are the *Association* instance the event occurred in
    #   and the *datetime.datetime* the event occurred at
    requestor = event.assoc.requestor
    timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    year = event.timestamp.year
    month = event.timestamp.month
    day = event.timestamp.day
    date = event.timestamp.date()
    msg = (
        "Received C-STORE service request from ({}, {}, {}) at {}"
        .format(requestor.ae_title, requestor.address, requestor.port, timestamp)
    )
    # logger.info(msg)

    print('  Get New DICOM  '.center(100,'='))
    print(msg)
    print(ds[0x008, 0x050]) # Accession Number
    print(ds[0x010, 0x020]) # Patient ID (HN)
    print(ds[0x008, 0x020]) # Study Date
    print(ds[0x008, 0x030]) # Study Time
    # Store metadata
    metadata_df = extract_dcm_info(ds)
    # Add extra info from event
    metadata_df.loc[0, 'ae_title'] = requestor.ae_title.strip()
    metadata_df.loc[0, 'address'] = requestor.address
    metadata_df.loc[0, 'port'] = requestor.port
    metadata_df.loc[0, 'Receiving_Time'] = str(timestamp)
    # create_path_and_save_dcm(ds, metadata_df)
    # create_path_and_save_png(ds, metadata_df)
    # print(ds)

    folder_path = os.path.join(BASE_DIR, 'resources', 'log', 'log_receive_dcm')
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    hn_map_path = os.path.join(folder_path, 'hn_map.csv')

    patientID = 0
    if not os.path.exists(hn_map_path):
        pd.DataFrame.from_records([{'ID': ds.PatientID}]).to_csv(hn_map_path, index=False)
        patientID = 1
    else:
        df = pd.read_csv(hn_map_path)
        patientID = df[df['ID'] == int(ds.PatientID)].index.tolist()
        if patientID == []:
            pd.DataFrame.from_records([{'ID': ds.PatientID}]).to_csv(hn_map_path, index=False, mode='a', header=False)
            df = pd.read_csv(hn_map_path)
            patientID = df[df['ID'] == int(ds.PatientID)].index.tolist()
        patientID = patientID[0] + 1
        del df

    metadata_df.loc[0, 'ID'] = patientID

    log_dir = f'{BASE_DIR}/resources/log/log_receive_dcm/{year}/{month}'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    path_store_log_receive_dcm = os.path.join(log_dir, str(date)+'.csv' )
    print(f'Save meta_data file at: {path_store_log_receive_dcm}')
    
    if not os.path.exists(path_store_log_receive_dcm):
        metadata_df.to_csv(path_store_log_receive_dcm, index=False)
    else:
        metadata_df.to_csv(path_store_log_receive_dcm, index=False, mode='a', header=False)

    # Avoid processing other X-ray part. (Focus only Chest X-ray)
    # prod_code_accept_list = ['R1201', 'R1203', 'R2201', 'R2202']
    # prod_code = ds[0x020,0x0010].value # Code Value of Procedure Code Sequence
    # if (prod_code not in prod_code_accept_list) | (ds.BodyPartExamined != 'CHEST'):
    try :
        BodyPartExamined = ds.BodyPartExamined
        # prod_code = ds.ViewPosition
    except:
        print("""'Dataset' object has no attribute 'BodyPartExamined'""")
        print('Skip this DICOM file.')
        print('  Done  '.center(100,'='))
        # folder_path = f'DICOMOBJ/NoBodyPartExamined/{year}/{month}/{day}'
        # Path(folder_path).mkdir(parents=True, exist_ok=True)
        # ds.save_as(f'{folder_path}/{ds.StudyDate}_{ds.PatientID}.dcm', write_like_original=False)
        clear_backup(path_backup_event)
        return 0x0000

    try: 
        if 'StudyDescription' in ds:
            StudyDescription = ds.StudyDescription
        elif 'ProcedureCodeSequence' in ds:
            StudyDescription = ds.ProcedureCodeSequence[0][('0008', '0104')].value
        elif 'PerformedProtocolCodeSequence' in ds:
            StudyDescription = ds.PerformedProtocolCodeSequence[0][('0008', '0104')].value
        elif 'RequestAttributesSequence' in ds:
            StudyDescription = ds.RequestAttributesSequence[0][('0040', '0007')].value
        else:
            StudyDescription = ''

    except:
        StudyDescription = ''

    # prod_code_accept_list = ['R1201', 'R1203', 'R2201', 'R2202', 'PA', 'AP']
    if  (BodyPartExamined != 'CHEST') | ('lateral' in StudyDescription.lower()): # filter non-chest and lateral view out
    # if  BodyPartExamined != 'CHEST':
        # print(f'Procedure Code "{prod_code}" not meet criteria({prod_code_accept_list}).')
        print(f'"{BodyPartExamined}" is not CHEST.')
        print(f'Or StudyDescription "{StudyDescription}" is lateral view.')
        print('Skip this DICOM file.')
        print('  Done  '.center(100,'='))
        clear_backup(path_backup_event)
        return 0x0000
    elif 'CR' not in Accession_Number:
        print('Not Found Accession Number')
        print('Skip this DICOM file.')
        print('  Done  '.center(100,'='))
        # folder_path = f'DICOMOBJ/NoAccessionNumber/{year}/{month}/{day}'
        # Path(folder_path).mkdir(parents=True, exist_ok=True)
        # ds.save_as(f'{folder_path}/{ds.StudyDate}_{ds.PatientID}.dcm', write_like_original=False)
        clear_backup(path_backup_event)
        return 0x0000
    else:
        # print(f'Procedure Code "{prod_code}" meet criteria({prod_code_accept_list})')
        print(f'"{BodyPartExamined}" is CHEST' )
        print(f'Or StudyDescription "{StudyDescription}" is not lateral view.')

    print(f'Receive DICOM and backup complete with execution time: {time.time() - start_time :.2f} seconds')

    # deleting and clear the variable from memory in python
    del ds, event, metadata_df

    os.rename(os.path.join(BACKUP_DIR, f'{Accession_Number}.evt'), os.path.join(BASE_DIR, 'resources', 'files', f'{Accession_Number}.evt'))

    return 0x0000 # Return a *Success* status

def start_server(host, port):

    print(f'Start the SCP on ({host}, {port}) in blocking mode')
    
    # Resend backup files    
    Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
    threading.Thread(target=startup_resend, args=(BACKUP_DIR, 30)).start()

    # Bind our C-STORE handler
    handlers = [(evt.EVT_C_STORE, handle_store)]
    
    # Start the SCP on (host, port) in blocking mode
    ae.start_server((host, port), block=True, evt_handlers=handlers)
    
def main() -> None:
    host = args.host
    port = args.port
    start_server(host, port)

if __name__ == "__main__":
    main()