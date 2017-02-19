from dataset import Dataset


class VisrDataset(Dataset):

    def __init__(self):
        super(VisrDataset, self).__init__("VISR")

    def download(self):
        print "Not yet implemented."

    def load(self, config=None):
        print "Not yet implemented."
