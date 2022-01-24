from pydicom.uid import ExplicitVRLittleEndian
from pynetdicom import AE, debug_logger, evt
from pynetdicom.sop_class import CTImageStorage

debug_logger()

def handle_store(event):
    """Handle EVT_C_STORE events."""
    return 0x0000

handlers = [(evt.EVT_C_STORE, handle_store)]

ae = AE()
ae.add_supported_context(CTImageStorage, ExplicitVRLittleEndian)
ae.add_supported_context('1.2.840.10008.1.1')
ae.start_server(("127.0.0.1", 11112), block=True, evt_handlers=handlers)