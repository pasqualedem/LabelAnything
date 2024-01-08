import torch
from torch.utils.data.sampler import BatchSampler

class VariableBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_sizes, num_examples):
        self.data_source = data_source
        self.batch_sizes_num_examples = zip(batch_sizes, num_examples)
        self.num_batches = len(batch_sizes)

        if self.num_batches == 0:
            raise ValueError("At least one batch size should be provided.")

    def __iter__(self):
        num_samples = len(self.data_source)
        indices = torch.randperm(num_samples, generator=torch.Generator()).tolist()

        for batch_size, num_examples in self.batch_sizes_num_examples:
            batch = []
            while len(batch) < batch_size and indices:
                batch.append((indices.pop(0), num_examples))
            yield batch
    
            
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        index, value = index
        print(value)
        return self.data[index]

    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":
    # Example usage:
    # Assuming you have a dataset named 'my_dataset'
    my_dataset = MyDataset(torch.randn((100, 3, 32, 32)))  # Example dataset, replace with your actual dataset

    # Specify a list of batch sizes
    batch_sizes_list = [2, 4, 3]
    num_examples = [10, 20, 30]

    # Create an instance of your custom batch sampler
    variable_batch_sampler = VariableBatchSampler(my_dataset, batch_sizes=batch_sizes_list, num_examples=num_examples)

    # Create a data loader using the custom batch sampler
    data_loader = torch.utils.data.DataLoader(my_dataset, batch_sampler=variable_batch_sampler)

    # Iterate over the data loader
    for batch in data_loader:
        print("Batch Size:", len(batch))
        # Your training logic here
