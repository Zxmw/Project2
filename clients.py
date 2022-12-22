import copy
import torch

from getData import GetDataSet
from torchvision import transforms
from torch.utils.data import DataLoader


class client(object):
    def __init__(self, trainDataSet, dev, client_id):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.id = client_id

    def localUpdate(
        self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters
    ):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(
            self.train_ds, batch_size=localBatchSize, shuffle=True, num_workers=4
        )
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                opti.zero_grad()
                loss.backward()
                opti.step()

        return Net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(
        self,
        dataSetName,
        data_set_path,
        input_image_height,
        input_image_width,
        isIID,
        numOfClients,
        dev,
        batchsize,
        test_frac,
    ):
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

    def dataSetBalanceAllocation(
        self, input_image_height, input_image_width, test_frac
    ):
        self.transform = transforms.Compose([transforms.ToTensor()])
        AllDataSet = GetDataSet(
            dataSetName=self.data_set_name,
            transform=self.transform,
            dataset_path=self.data_set_path,
            batch_size=self.batch_size,
            isIID=self.is_iid,
            test_frac=test_frac,
            input_image_height=input_image_height,
            input_image_width=input_image_width,
        )

        test_set = AllDataSet.test_set
        pin_memory_flag = True if torch.cuda.is_available() else False
        self.test_data_loader = DataLoader(
            test_set,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=pin_memory_flag,
            num_workers=4,
        )

        train_set = AllDataSet.train_set

        local_data_size = AllDataSet.train_data_size // self.num_of_clients
        all_train_img_ls = train_set.img_name_ls
        all_train_label_ls = train_set.label_name_ls
        for i in range(self.num_of_clients):

            local_data = copy.deepcopy(train_set)
            local_data.set_img_name_ls(
                all_train_img_ls[i *
                                 local_data_size: (i + 1) * local_data_size],
                all_train_label_ls[i *
                                   local_data_size: (i + 1) * local_data_size],
            )

            someone = client(local_data, self.dev, i)
            self.clients_set["client{}".format(i)] = someone
