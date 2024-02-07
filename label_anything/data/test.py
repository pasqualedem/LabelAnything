from torch.utils.data import Dataset


class LabelAnythingTestDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
    
    def extract_prompts(self):
        raise NotImplementedError()