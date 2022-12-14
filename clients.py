import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
from PIL import Image
from torchvision import transforms
import copy
import os.path
import cv2


class client(object):
    def __init__(self, trainDataSet, dev, client_id):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.id = client_id

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(
            self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, dataSetName, data_set_path, input_image_height, input_image_width, isIID, numOfClients, dev, batchsize, test_frac):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.batch_size = batchsize
        self.test_data_loader = None
        self.data_set_path = data_set_path

        self.dataSetBalanceAllocation(
            input_image_height, input_image_width, test_frac)

    def dataSetBalanceAllocation(self, input_image_height, input_image_width, test_frac):
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((input_image_height,
                                                                input_image_width), interpolation=Image.NEAREST),
                                             transforms.ToTensor()])
        AllDataSet = GetDataSet(dataSetName=self.data_set_name, transform=self.transform,
                                dataset_path=self.data_set_path, batch_size=self.batch_size, isIID=self.is_iid,)

        test_set = AllDataSet.test_set
        pin_memory_flag = True if torch.cuda.is_available() else False
        self.test_data_loader = DataLoader(
            test_set, shuffle=False, batch_size=self.batch_size, pin_memory=pin_memory_flag)

        train_set = AllDataSet.train_set

        local_data_size = AllDataSet.train_data_size // self.num_of_clients
        all_train_img_ls = train_set.img_name_ls
        '''shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)'''
        for i in range(self.num_of_clients):
            '''shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)'''
            local_data = copy.deepcopy(train_set)
            local_data.set_img_name_ls(
                all_train_img_ls[i*local_data_size:(i+1)*local_data_size])

            someone = client(local_data, self.dev, i)
            self.clients_set['client{}'.format(i)] = someone


if __name__ == "__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])
