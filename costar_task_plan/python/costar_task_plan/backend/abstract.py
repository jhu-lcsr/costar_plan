

class AbstractBackend(object):
    '''
    Package lookup and communication tools, posed in a general way.
    '''

    def findPackage(self, name):
        raise RuntimeError('backend functionality not implemented')

_backend = None

def FindPackage(self, pkg):
    '''
    '''
    return _backend.findPackage(pkg)

def SetBackend(self, backend):
    '''
    Reset current backend
    '''
    _backend = backend
