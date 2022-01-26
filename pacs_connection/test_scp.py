from pydicom.uid import ExplicitVRLittleEndian
from pynetdicom import AE, debug_logger, evt, ALL_TRANSFER_SYNTAXES
from pynetdicom.sop_class import CTImageStorage, DigitalXRayImageStorageForPresentation

debug_logger()

def handle_store(event):
    """Handle EVT_C_STORE events."""
    return 0x0000

handlers = [(evt.EVT_C_STORE, handle_store)]

ae = AE()
ae.add_supported_context(CTImageStorage, ExplicitVRLittleEndian)
ae.add_supported_context(DigitalXRayImageStorageForPresentation, ALL_TRANSFER_SYNTAXES)
ae.start_server(("127.0.0.1", 11113), block=True, evt_handlers=handlers)