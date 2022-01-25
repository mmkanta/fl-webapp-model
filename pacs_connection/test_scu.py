from pynetdicom import AE

# PACS server
AE_title = b'SYNAPSEDICOM'
addr = "127.0.0.1"
port = 11112

ae = AE(ae_title=AE_title)
ae.add_requested_context('1.2.840.10008.1.1')

assoc = ae.associate(addr, port)

if assoc.is_established:
    status = assoc.send_c_echo()

    print('status:', status)
    if status:
        print('C-ECHO Response: 0x{0:04x}'.format(status.Status))

    assoc.release()
else:
    print('Fail')