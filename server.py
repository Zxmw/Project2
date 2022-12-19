import os
import torch
import logging
import argparse
import numpy as np

from tqdm import tqdm
from Models import UNet
from torch import optim
from clients import ClientsGroup, client
from utils import DiceLoss, SegmentationMetric


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg"
)
parser.add_argument(
    "-g", "--gpu", type=str, default="0", help="gpu id to use(e.g. 0,1,2,3)"
)
parser.add_argument(
    "-nc", "--num_of_clients", type=int, default=1, help="numer of the clients"
)
parser.add_argument(
    "-cf",
    "--cfraction",
    type=float,
    default=1,
    help="C fraction, 0 means 1 client, 1 means total clients",
)
parser.add_argument("-E", "--epoch", type=int,
                    default=1, help="local train epoch")
parser.add_argument(
    "-B", "--batchsize", type=int, default=1, help="local train batch size"
)
parser.add_argument(
    "-mn", "--model_name", type=str, default="UNet", help="the model to train"
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=0.0001,
    help="learning rate, \
                    use value from origin paper as default",
)
parser.add_argument(
    "-vf",
    "--val_freq",
    type=int,
    default=5,
    help="model validation frequency(of communications)",
)
parser.add_argument(
    "-sf",
    "--save_freq",
    type=int,
    default=20,
    help="global model save frequency(of communication)",
)
parser.add_argument(
    "-ncomm", "--num_comm", type=int, default=20, help="number of communications"
)
parser.add_argument(
    "-sp",
    "--save_path",
    type=str,
    default="./checkpoints",
    help="the saving path of checkpoints",
)
parser.add_argument(
    "-lp",
    "--log_path",
    type=str,
    default="./logs",
    help="the saving path of logs",
)
parser.add_argument(
    "-iid", "--IID", type=int, default=1, help="the way to allocate data to clients"
)

parser.add_argument(
    "-dname",
    "--dataset_name",
    type=str,
    default="AsphaltCrack",
    help="the name of the dataset",
)
parser.add_argument(
    "-tf", "--test_frac", type=float, default=0.15, help="the fraction of the test data"
)
parser.add_argument(
    "-imgw",
    "--image_width",
    type=int,
    default=480,
    help="the width of most input images",
)
parser.add_argument(
    "-imgh",
    "--image_height",
    type=int,
    default=320,
    help="the height of most input images",
)


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args["save_path"])

    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net = None
    seed = 1  # seed必须是int，可以自行设置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
    torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
    np.random.seed(seed)  # numpy产生的随机数一致

    if args["model_name"] == "UNet":
        net = UNet()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = DiceLoss()
    opti = optim.Adam(net.parameters(), lr=args["learning_rate"])
    dataset_path = None
    if args["dataset_name"] == "AsphaltCrack":
        dataset_path = "./SematicSeg_Dataset"
    elif args["dataset_name"] == "ConcreteCrack":
        dataset_path = "./concreteCrackSegmentationDataset"
    myClients = ClientsGroup(
        dataSetName=args["dataset_name"],
        data_set_path=dataset_path,
        input_image_height=args["image_height"],
        input_image_width=args["image_width"],
        isIID=args["IID"],
        numOfClients=args["num_of_clients"],
        dev=dev,
        batchsize=args["batchsize"],
        test_frac=args["test_frac"],
    )
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args["num_of_clients"] * args["cfraction"], 1))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(
                    args["log_path"],
                    args["dataset_name"]
                    + "-"
                    + str(args["num_of_clients"])
                    + "-"
                    + str(args["cfraction"])
                    + ".log",
                )
            ),
            logging.StreamHandler(),
        ],
    )
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    test_loss_list = []
    for i in range(args["num_comm"]):
        print("communicate round {}".format(i + 1))

        order = np.random.permutation(args["num_of_clients"])
        clients_in_comm = ["client{}".format(i) for i in order[0:num_in_comm]]
        # metric
        totalTestLoss = 0
        mIOU = 0
        metric = SegmentationMetric(2)
        sum_parameters = None

        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(
                args["epoch"],
                args["batchsize"],
                net,
                loss_func,
                opti,
                global_parameters,
            )
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + \
                        local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = sum_parameters[var] / num_in_comm

        with torch.no_grad():
            # if (i + 1) % args['val_freq'] == 0:
            net.load_state_dict(global_parameters, strict=True)
            # set the model in evaluation mode
            net.eval()
            iter_num = 0
            # loop over the validation set
            for (x, y) in testDataLoader:
                iter_num += 1
                # send the input to the device
                (x, y) = (x.to(dev), y.to(dev))
                # make the predictions and calculate the validation loss
                pred = net(x)
                # # plot the prediction and ground truth mask
                # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                # ax[0].imshow(x[0].cpu().permute(1, 2, 0))
                # ax[0].set_title("Input Image")
                # # ax[1].imshow(pred[0].cpu().squeeze(), cmap="gray")
                # ax[1].imshow(
                #     torch.sigmoid(pred[0]).cpu().squeeze(), cmap="gray", vmin=0, vmax=1
                # )
                # ax[1].set_title("Predicted Mask")
                # plt.savefig(os.path.join(
                #     "./plots", "pred_{}.png".format(iter_num)))
                totalTestLoss += loss_func(pred, y)
                # evalution metric
                metric.addBatch(pred, y)
        # print(metric.confusionMatrix)
        # output metric.confusionMatrix to log file
        logging.info("confusionMatrix:{}".format(metric.confusionMatrix))
        avgmIOU = metric.meanIntersectionOverUnion()
        avgTestLoss = totalTestLoss / iter_num
        precison = metric.precision()[1]
        recall = metric.recall()[1]
        F1score = metric.F1score()[1]
        test_loss_list.append(avgTestLoss.cpu().detach().numpy())
        # print(
        #     "Test loss: {:.4f},mIOU:{:.2f},precision:{:.2f},recall:{:.2f},F1score:{:.2f}".format(
        #         avgTestLoss, avgmIOU, precison, recall, F1score
        #     )
        # )
        logging.info(
            "Test loss: {:.4f},mIOU:{:.2f},precision:{:.2f},recall:{:.2f},F1score:{:.2f}".format(
                avgTestLoss, avgmIOU, precison, recall, F1score
            )
        )
        if (i + 1) % args["save_freq"] == 0:
            torch.save(
                net,
                os.path.join(
                    args["save_path"],
                    "{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}".format(
                        args["model_name"],
                        i,
                        args["epoch"],
                        args["batchsize"],
                        args["learning_rate"],
                        args["num_of_clients"],
                        args["cfraction"],
                    ),
                ),
            )
