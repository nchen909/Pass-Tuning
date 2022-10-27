import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):

    def __getitem__(self, index):

        return np.random.randint( 0, 1000, 3)

    def __len__(self):

        return 30

dataset = RandomDataset

dataloader = DataLoader(dataset, batch_size= 3, num_workers= 5, pin_memory=True)
print(dataloader)
print(len(dataloader))
bar = tqdm(dataloader, total=len(dataloader), desc="Training")
for step,batch in enumarate(bar):
    print(len(batch))
    print(batch)
