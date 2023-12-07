#importing libraries

import os
from typing import Tuple
from zipfile import ZipFile
from PIL import Image

from numpy import array
from pandas import read_csv
from requests import get
from torch.utils.data import Dataset

#Dataset Class

class HotDogDataset(Dataset):

    #Child class of torch.utils.data.Dataset.

    #This is a wrapper for mapping from hotdog/not hotdog images to the target.


    def __init__(self, dir_name, transform=None) -> None:

        #Initialise a HotdogDataset class

        #param dir_name: The name of the folder holding the data.
        #param transform:

        self.transform = transform
        # extract the data if needed, or can use the folder directly if extracted before
        if not os.path.isdir(os.path.join(os.getcwd(), f"{dir_name}")):
            with ZipFile(os.path.join(os.getcwd(), f"{dir_name}.zip"), 'r') as zip_ref:
                zip_ref.extractall("./")
                zip_ref.close()
        # Load the metadata.
        self.data = read_csv(os.path.join(os.getcwd(), dir_name, f"{dir_name}_labels.csv"))
        # Number of classes.
        self.n_classes = len(self.data['y'].unique())

    def __len__(self) -> int:

        #return: The length of the training/testing dataset.

        return len(self.data)

    def __getitem__(self, idx) -> Tuple[array, str]:

        #Return the input and target at a specific index of the dataset.

        #:param idx: The index of the data to be returned.
        #:return: Key-value pair at the specified index.

        # Open the corresponding Image.
        image = Image.open(os.path.join(os.getcwd(), self.data.loc[idx, 'file_name']))
        # Retrieve the label.
        y = self.data.loc[idx, 'y']
        # Transform the image if necessary.
        if self.transform is not None:
            image_ = self.transform(image)
            image.close()
        else:
            image_ = array(image)
            image.close()
        return image_, y
