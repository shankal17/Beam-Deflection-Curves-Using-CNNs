import os
import torch
import numpy as np

from natsort import natsorted
from torch.utils.data import Dataset

class BeamDataset(Dataset):
    """Dataset class for deflected beams
    ...
    
    Attributes
    ----------
    data_folder : String
        Path of folder containing deflected beam information
    total_beams : List
        List of sorted deflected beam path names
    split_type : String
        Descriptive of type of dataset
    
    Methods
    -------
    __getitem__(idx)
        Returns beam information at index idx
    __len__()
        Returns how many beams are in the dataset
    """

    def __init__(self, data_folder, split_type):
        self.data_folder = data_folder
        all_beams = os.listdir(data_folder)
        self.total_beams = natsorted(all_beams)
        self.split_type = split_type.lower()
        assert self.split_type in {'train', 'test', 'val'}

    def __getitem__(self, idx):
        """Returns beam information at index idx

        Parameters
        ----------
        idx : int
            Index of beam file path
        
        Returns
        -------
        torch.Tensor
            Deflected beam information
        """

        beam_loc = os.path.join(self.data_folder, self.total_beams[idx])
        loaded_beam = np.load(beam_loc)
        loaded_beam = torch.from_numpy(loaded_beam)

        return loaded_beam

    def __len__(self):
        """Returns number of beams in the dataset

        
        Returns
        -------
        int
            Number of beams in the dataset
        """

        return len(self.total_beams)


if __name__ == '__main__':
    # Test to make sure that the dataset can be instantiated
    data = BeamDataset('data/train', 'train')
    print(data)
    print(data.__len__())
