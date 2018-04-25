
class GPUException(Exception):
    
    def __init__(self, message):
        
        super(GPUException, self).__init__(message=message)