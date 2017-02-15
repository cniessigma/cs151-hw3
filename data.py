class Data:
    """ Simple class for data files """
    fdir = "data/"
    training = "hw3train.txt"
    testing = "hw3test.txt"
    validate = "hw3validation.txt"
    vector_size = 22

    def __init__(self, override = None):
        if override != None:
            self.fdir = override['dir']
            self.training = override['training']
            self.testing = override['testing']
            self.validate = override['validate']
            self.projection = override['projection']
            self.vector_size = override['size']