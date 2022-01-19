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