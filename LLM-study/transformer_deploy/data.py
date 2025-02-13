import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data_lenth_max = 6
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, _):
        data_length = torch.randint(low = 1, high = self.data_lenth_max + 1, size = (1, )).item()
        data = torch.randint(low = 0, high = 9, size = (data_length, ))
        data = F.pad(data, (0, self.data_lenth_max - data.shape[0]))
        label = data.clone().detach()
        return data, label
    
if __name__ == "__main__":
    num_samples = 1000
    dataset = CustomDataset(num_samples = num_samples)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
    
    for batch in dataloader:
        data, label = batch
        print(">>>>>>>>>>>>>>>>>")
        print("Data: ", data)
        print("Label: ", label)
        
        for d, l in zip(data, label):
            assert torch.equal(d, l), f"data element {d} is not equal to label element {l}"