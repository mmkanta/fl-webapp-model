from pydicom.dataset import Dataset

from pynetdicom import AE, debug_logger
# from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelFind
from pynetdicom.sop_class import DigitalXRayImagePresentationStorage
debug_logger()

from pydicom.uid import JPEG2000Lossless, JPEGLosslessSV1

addr = 'ADDR'
port = 'PORT'
ae_title = '??'

# https://pydicom.github.io/pynetdicom/stable/reference/generated/pynetdicom._handlers.doc_handle_find.html#pynetdicom._handlers.doc_handle_find
# https://pydicom.github.io/pynetdicom/stable/examples/qr_find.html
def c_find(patient_id):
    ae = AE()
    transfer_syntaxes = [JPEGLosslessSV1]
    ae.add_requested_context(DigitalXRayImagePresentationStorage, transfer_syntax=transfer_syntaxes)

    # Create our Identifier (query) dataset
    ds = Dataset()
    ds.SOPClassesInStudy = ''
    ds.PatientID = patient_id
    ds.StudyInstanceUID = ''
    ds.QueryRetrieveLevel = 'STUDY'

    # Associate with the peer AE at IP 127.0.0.1 and port 11112
    assoc = ae.associate(addr, port, ae_title)
    if assoc.is_established:
        # Send the C-FIND request
        responses = assoc.send_c_find(ds, DigitalXRayImagePresentationStorage)
        for (status, identifier) in responses:
            if status:
                print('C-FIND query status: 0x{0:04X}'.format(status.Status))
            else:
                print('Connection timed out, was aborted or received invalid response')

        # Release the association
        assoc.release()
    else:
        print('Association rejected, aborted or never connected')