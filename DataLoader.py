import numpy as np

class DataLoader:
    def __init__(self,inputs,labels,batch_size=1,shuffle=False,drop_last=False):
        self.inputs = inputs
        self.labels = labels
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.drop_last = drop_last
        self.indices = 0
        self.number_of_batches = 0

    def __iter__(self):
        self.indices = np.arange(len(self.inputs))
        if(self.shuffle==True):
            np.random.shuffle(self.indices)
        self.indices = np.array_split(self.indices, np.arange(self.batch_size, len(self.indices), self.batch_size))
        if(self.drop_last==True and len(self.indices[-1])!=self.batch_size):
            self.indices.pop()
        self.number_of_batches = len(self.indices)
        for i in range(self.number_of_batches):
            yield self.inputs[self.indices[i],:], self.labels[self.indices[i],:]
