import numpy as np
import gzip
import os
import cv2
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split



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
        elif self.name == "AsphaltCrack":
            self.train_set, self.test_set = self.AsphaltCrackDataSetConstruct(
                isIID, test_split=test_frac
            )

    def DataSetConstruct(
        self,
        isIID,
        image_document_name,
        label_document_name,
        test_split=0.15,
    ):
        image_dataset_path = os.path.join(self.dataset_path, image_document_name)
        mask_dataset_path = os.path.join(self.dataset_path, label_document_name)
        imagePaths = sorted(list(os.listdir(image_dataset_path)))
        maskPaths = sorted(list(os.listdir(mask_dataset_path)))

        # get first 10 images and masks
        imagePaths = imagePaths[:10]
        maskPaths = maskPaths[:10]
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
        base_output = "tmp"

        # define the path to the output serialized model, model training
        # plot, and testing image paths
        test_image_path = os.path.sep.join([base_output, "test_image_paths.txt"])
        test_mask_path = os.path.sep.join([base_output, "test_mask_paths.txt"])

        print("[INFO] saving testing image paths...")
        f = open(test_image_path, "w")
        f.write("\n".join(testImages))
        f.close()

        f = open(test_mask_path, "w")
        f.write("\n".join(testMasks))
        f.close()

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

        # create the train and test datasets
        if isIID:
            order = np.arange(len(trainImages))
            np.random.shuffle(order)
            trainImages = np.array(trainImages)[order]
            trainMasks = np.array(trainMasks)[order]
        else:
            num_white = []
            for i in range(trainDS.data_len):
                mask = trainDS[i][1]
                mask = mask.numpy()
                total_white = np.sum(mask)
                print(total_white)
                num_white.append(total_white)

            sorted_id = sorted(range(trainDS.data_len), key=lambda k: num_white[k])
            trainImages = np.array(trainImages)[sorted_id]
            trainMasks = np.array(trainMasks)[sorted_id]

        trainDS.set_img_name_ls(trainImages, trainMasks)
        print(f"[INFO] found {len(trainDS)} examples in the training set...")
        print(f"[INFO] found {len(testDS)} examples in the test set...")
        # create the training and test data loaders
        return trainDS, testDS

    def ConcreteCrackDataSetConstruct(self, isIID, test_split=0.15):
        return self.DataSetConstruct(isIID, "rgb", "BW", test_split)

    def AsphaltCrackDataSetConstruct(self, isIID, test_split=0.15):
        return self.DataSetConstruct(isIID, "Original Image", "Labels", test_split)


if __name__ == "__main__":
    "test data set"
    transform = transforms.Compose([transforms.ToTensor()])
    ConcreteCrackDataSet = GetDataSet(
        "ConcreteCrack",
        dataset_path="./concreteCrackSegmentationDataset",
        transform=transform,
        isIID=False,
    )  # test NON-IID

    trainDS = ConcreteCrackDataSet.train_set
    for i in range(len(trainDS)):
        img, label = trainDS[i]
        if i % 10 == 0:
            fix, ax = plt.subplots(1, 2)
            ax[0].imshow(img.permute(1, 2, 0))
            ax[1].imshow(label.squeeze(), cmap="gray")
            plt.savefig(f"./plottest/{i}.png")
