import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, sequence_fasta, binding_site_fasta):
        self.sequences = []
        self.binding_sites = []

        with open(sequence_fasta, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    continue
                self.sequences.append(line.strip())

        with open(binding_site_fasta, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    continue
                raw_sites = line.strip()
                sites = [int(x) for x in raw_sites.split()]
                self.binding_sites.append(sites)

        assert len(self.sequences) == len(self.binding_sites), "same number of sequences and binding site annotations required"
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        binding_site_indices = self.binding_sites[idx]

        labels = [0] * len(sequence)
        for index in binding_site_indices:
            if index < len(labels):
                labels[index] = 1
        sample = {'sequence': sequence, 'labels': torch.tensor(labels, dtype=torch.long)}
        return sample