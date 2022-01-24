import pandas as pd
from pydicom import dcmread
from pydicom.uid import ImplicitVRLittleEndian

from pynetdicom import AE, VerificationPresentationContexts, StoragePresentationContexts
from pynetdicom.sop_class import DigitalXRayImageStorageForPresentation

from pydicom.uid import JPEG2000Lossless, JPEGLosslessSV1
from pynetdicom import AE, StoragePresentationContexts, DEFAULT_TRANSFER_SYNTAXES

import argparse

from pynetdicom import debug_logger
# Setup logging to use the StreamHandler at the debug level
debug_logger()

from datetime import datetime, timedelta
from pathlib import Path
import os
import time

# https://zetcode.com/python/argparse/
parser = argparse.ArgumentParser()

parser.add_argument('-f', '--file', type=str, help="dicom file path")
parser.add_argument('-a', '--ae_title_scp', type=str, default='', help="AE Title SCP")
parser.add_argument('-s', '--host', type=str, default='', help="host name")
parser.add_argument('-p', '--port', type=int, default=11112, help="destination port number")

args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def SCU(path_dcm, addr, port, ae_title_scp = b'ANY-SCP'):
    # Date time
    date_time = datetime.now()
    year = date_time.year
    month = date_time.month
#     day = date_time.day
    date = date_time.date()
    yesterday = date - timedelta(days=1)

    # Read target DICOM file
    ds = dcmread(path_dcm)
    # Extract the metadata
    metadata_dict = {}
    metadata_dict["Accession Number"] = ds.AccessionNumber
    metadata_dict["ID"] = ds.PatientID
    # metadata_dict["ae_title"] = ae_title_scp
    # metadata_dict["address"] = addr
    # metadata_dict["port"] = port
    metadata_dict["MediaStorageSOPClassUID_HM"] = ds.file_meta.__getitem__('MediaStorageSOPClassUID').repval
    metadata_dict["MediaStorageSOPInstanceUID_HM"] = ds.file_meta.__getitem__('MediaStorageSOPInstanceUID').repval
    metadata_dict["TransferSyntaxUID_HM"] = ds.file_meta.__getitem__('TransferSyntaxUID').repval
    metadata_dict["Study Date"] = ds.StudyDate
    metadata_dict["Study Time"] = ds.StudyTime
    metadata_dict["Sending_Time"] = str(date_time)

    # https://pydicom.github.io/pynetdicom/stable/reference/generated/pynetdicom.ae.ApplicationEntity.html#pynetdicom.ae.ApplicationEntity.associate
    ae = AE(ae_title=b'PYNETDICOM')
    transfer_syntaxes = [JPEGLosslessSV1] # ['1.2.840.10008.1.2.4.70'] # JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])
    ae.add_requested_context(DigitalXRayImageStorageForPresentation, transfer_syntax= transfer_syntaxes) # Digital X-Ray Image Storage - For Presentation
    assoc = ae.associate(addr, port, ae_title = ae_title_scp)

    # Define path of log_send_c_store from today and yesterday 
    folder_path = f'{BASE_DIR}/resources/log/log_send_c_store/{year}/{month}'
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    path_store_log_send_c_store_today = os.path.join(folder_path, str(date)+'.csv' )
    folder_path = f'{BASE_DIR}/resources/log/log_send_c_store/{yesterday.year}/{yesterday.month}'
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    path_store_log_send_c_store_yesterday = os.path.join(folder_path, str(yesterday)+'.csv' )

    # with open('database/Log_Accession_Number.txt', 'r') as f:
    #     log_acc_num = f.read().split('\n')

    # Check whether this accession number was already sent. If yes, change SOPInstanceUID Back to previous value.
    try:
        df_log_send_c_store = pd.read_csv(path_store_log_send_c_store_yesterday,error_bad_lines=False).append(pd.read_csv(path_store_log_send_c_store_today,error_bad_lines=False)).reset_index(drop=True)
    except:
        print('Cannot read path_store_log_send_c_store_yesterday or path_store_log_send_c_store_today')
        df_log_send_c_store = pd.DataFrame()

    if 'Accession Number' in df_log_send_c_store.columns:
        target_acc_num = df_log_send_c_store.loc[df_log_send_c_store['Accession Number'] == ds.AccessionNumber]
    else:
        target_acc_num = []
    if len(target_acc_num) > 0:
        target_acc_num = target_acc_num.dropna(subset=['MediaStorageSOPInstanceUID_HM'])
        target_acc_num = target_acc_num.iloc[-1] # get the latest record
        print(f"{ds.AccessionNumber} has already been sent.\nChange SOPInstanceUID back to previous value ({target_acc_num['MediaStorageSOPInstanceUID_HM']})")
        ds.file_meta.MediaStorageSOPInstanceUID = target_acc_num['MediaStorageSOPInstanceUID_HM']
        ds.SOPInstanceUID = target_acc_num['MediaStorageSOPInstanceUID_HM']
        metadata_dict["MediaStorageSOPInstanceUID_HM"] = ds.file_meta[(0x002, 0x0003)].value #ds.file_meta.__getitem__('MediaStorageSOPInstanceUID').repval


    # Sending Process
    print(f"Sending {ds.AccessionNumber} DICOM file .....")
    n_try = 1
    if assoc.is_established:
        # Use the C-STORE service to send the dataset
        # returns the response status as a pydicom Dataset
        status = assoc.send_c_store(ds)
        print("Sending Time:", date_time.ctime())
        # Check the status of the storage request
        while (not status) or (status.Status != 0): # "not status" can only check is association timeout
            if n_try >= 5:
                print(f"{ds.AccessionNumber} Try #{n_try} Failed.\nSkip sending this DICOM file")
                break
            else:
                print(f"{ds.AccessionNumber} Try #{n_try} Failed.\nSleep for 2 seconds and re-send_c_store again")
                time.sleep(2)
                assoc.release()
                # The association with a peer SCP must be established (again) before sending a C-STORE request
                assoc = ae.associate(addr, port, ae_title = ae_title_scp)
                status = assoc.send_c_store(ds)
                n_try += 1

        if status:
            # If the storage request succeeded this will be 0x0000
            print('C-STORE request status: 0x{0:04x} with #{1} try'.format(status.Status, n_try))
            with open('log/Log_Accession_Number.txt', 'a') as f:
                f.write(str(ds.AccessionNumber) + '\n')
        else:
            print('Connection timed out, was aborted or received invalid response')
        
        print(status)
        print(type(status), bool(status))
        print(status.Status)

        assoc.release()
        # print(f"Send DICOM file {path_dcm} complete")
        metadata_dict["Status"] = '0x{0:04x}'.format(status.Status)
        # print(metadata_dict)
        # print("status", status) # (0000, 0900) Status                              US: 0
        # print("status.Status", status.Status) # 0
        # print('status format', f'0x{0:04x}'.format(status.Status)) # 0x0000
    else:
        # Association rejected, aborted or never connected
        print('Failed to associate')
        # print('Association rejected, aborted or never connected')
        metadata_dict["Status"] = "Failed to associate"

    metadata_dict["n_try"] = n_try
    # Store log file to csv
    metadata_df = pd.DataFrame.from_records([metadata_dict])
    
    if not os.path.exists(path_store_log_send_c_store_today):
        metadata_df.to_csv(path_store_log_send_c_store_today, index=False)
    else:
        metadata_df.to_csv(path_store_log_send_c_store_today, index=False, mode='a', header=False)


    # # Store log_send_c_store single record to temp
    # folder_path = f'database/temp'
    # Path(folder_path).mkdir(parents=True, exist_ok=True)
    # metadata_df.to_csv(f'{folder_path}/{ds.AccessionNumber}.csv', index=False)

    # Delete DICOM after sent
    os.remove(path_dcm)
    
def main() -> None:
    path_dcm = args.file
    ae_title_scp = args.ae_title_scp
    host = args.host
    port = args.port
    SCU(path_dcm, host, port, ae_title_scp = ae_title_scp)

if __name__ == "__main__":
    main()

## example cmd: python SCU.py -a MY_ECHO_SCP -s localhost -p 11112 -f "/Users/plotpro/Downloads/Sample_Dicom_Present_ID/0034081.dcm"