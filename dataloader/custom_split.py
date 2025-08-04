from torch.utils.data import Subset

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx, sub_idx=None):
        actual_idx = self.indices[idx]
        return self.dataset.__getitem__(actual_idx, sub_idx=sub_idx)
