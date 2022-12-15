import numpy as np
import gzip
import os
import platform
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import cv2


class SegmentationDataSet(Dataset):
    def __init__(
        self,
        rootdir,
        labeldir,
        img_name_ls,
        label_name_ls,
        transform=None,
        input_image_height=320,
        input_image_width=480,
    ):
        assert len(img_name_ls) == len(label_name_ls)
        self.data_dir = rootdir
        self.label_dir = labeldir
        self.img_name_ls = img_name_ls
        self.label_name_ls = label_name_ls
        self.data_len = len(img_name_ls)
        self.transform = transform
        self.input_image_height = input_image_height
        self.input_image_width = input_image_width

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_name = self.img_name_ls[index]
        label_name = self.label_name_ls[index]
        img_path = os.path.join(self.data_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        img = cv2.imread(img_path)
        label = cv2.imread(label_path)
        # resize the image and mask
        img = cv2.resize(img, (self.input_image_width, self.input_image_height))
        label = cv2.resize(label, (self.input_image_width, self.input_image_height))
        # convert mask to 0 and 1
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label[label > 0] = 1
        label = label.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
        return img, label

    def set_img_name_ls(self, new_img_name_ls, new_label_name_ls):
        assert len(new_img_name_ls) == len(new_label_name_ls)
        self.img_name_ls = new_img_name_ls
        self.label_name_ls = new_label_name_ls
        self.data_len = len(new_img_name_ls)


class GetDataSet(object):
    def __init__(
        self,
        dataSetName,
        transform,
        dataset_path,
        batch_size=64,
        isIID=True,
        test_frac=0.15,
    ):
        self.name = dataSetName
        self.train_set = None
        self.test_set = None
        self.transform = transform
        self._index_in_train_epoch = 0
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.train_data_size = None

        if self.name == "ConcreteCrack":
            self.train_set, self.test_set = self.ConcreteCrackDataSetConstruct(
                isIID, test_split=test_frac
            )
        else:
            pass

    """def old_func(self,isIID):
        data_dir = r'.\data\MNIST'
        # data_dir = r'./data/MNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]



        self.test_data = test_images
        self.test_label = test_labels"""

    def ConcreteCrackDataSetConstruct(self, isIID, test_split=0.15):
        image_dataset_path = os.path.join(self.dataset_path, "rgb")
        mask_dataset_path = os.path.join(self.dataset_path, "BW")
        imagePaths = sorted(list(os.listdir(image_dataset_path)))
        maskPaths = sorted(list(os.listdir(mask_dataset_path)))

        # partition the data into training and testing splits using 85% of
        # the data for training and the remaining 15% for testing
        split = train_test_split(
            imagePaths, maskPaths, test_size=test_split, random_state=42
        )

        # unpack the training and testing image and mask paths
        (trainImages, testImages) = split[:2]
        (trainMasks, testMasks) = split[2:]

        self.train_data_size = len(trainImages)
        # write the testing image paths to disk so that we can use then
        # when evaluating/testing our model
        # define the path to the base output directory
        base_output = "output"

        # define the path to the output serialized model, model training
        # plot, and testing image paths
        self.model_path = os.path.join(base_output, "unet_tgs_salt.pth")
        self.plot_path = os.path.sep.join([base_output, "plot.png"])
        test_image_path = os.path.sep.join([base_output, "test_image_paths.txt"])
        test_mask_path = os.path.sep.join([base_output, "test_mask_paths.txt"])

        print("[INFO] saving testing image paths...")
        f = open(test_image_path, "w")
        f.write("\n".join(testImages))
        f.close()

        f = open(test_mask_path, "w")
        f.write("\n".join(testMasks))
        f.close()

        # create the train and test datasets
        if isIID:
            order = np.arange(len(trainImages))
            np.random.shuffle(order)
            trainImages = np.array(trainImages)[order]
            trainMasks = np.array(trainMasks)[order]
        else:
            raise NotImplementedError
            """labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]"""

        trainDS = SegmentationDataSet(
            image_dataset_path,
            mask_dataset_path,
            trainImages,
            trainMasks,
            transform=self.transform,
        )
        testDS = SegmentationDataSet(
            image_dataset_path,
            mask_dataset_path,
            testImages,
            testMasks,
            transform=self.transform,
        )
        print(f"[INFO] found {len(trainDS)} examples in the training set...")
        print(f"[INFO] found {len(testDS)} examples in the test set...")
        # create the training and test data loaders
        return trainDS, testDS

        """pin_memory_flag = True if torch.cuda.is_available() else False
        trainLoader = DataLoader(trainDS, shuffle=True,
            batch_size=self.batch_size, pin_memory=pin_memory_flag)
        testLoader = DataLoader(testDS, shuffle=False,
            batch_size=self.batch_size, pin_memory=pin_memory_flag)"""


if __name__ == "__main__":
    "test data set"
    mnistDataSet = GetDataSet("mnist", True)  # test NON-IID
    if (
        type(mnistDataSet.train_data) is np.ndarray
        and type(mnistDataSet.test_data) is np.ndarray
        and type(mnistDataSet.train_label) is np.ndarray
        and type(mnistDataSet.test_label) is np.ndarray
    ):
        print("the type of data is numpy ndarray")
    else:
        print("the type of data is not numpy ndarray")
    print("the shape of the train data set is {}".format(mnistDataSet.train_data.shape))
    print("the shape of the test data set is {}".format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])
