import argparse
import json
import time

import torch

# from dataset.oltp_dataset.oltp_utils import OLTPDataSet
# from dataset.terrier_tpch_dataset.terrier_utils import TerrierTPCHDataSet
from model_arch import QPPNet
from pg_utils import PostgresDataSet

parser = argparse.ArgumentParser(description="QPPNet Arg Parser")

# Environment arguments required

parser.add_argument(
    "--data_dir", type=str, default="./res_by_temp/", help="Dir containing train data"
)

parser.add_argument(
    "--dataset",
    type=str,
    default="POSTGRES",
    help="Select dataset [POSTGRES]",
)

parser.add_argument("--test_time", action="store_true", help="if in testing mode")

parser.add_argument(
    "--save_dir",
    type=str,
    default="./saved_model",
    help="Dir to save model weights (default: ./saved_model)",
)

parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
)

parser.add_argument("--scheduler", action="store_true")
parser.add_argument(
    "--step_size",
    type=int,
    default=1000,
    help="step_size for StepLR scheduler (default: 1000)",
)

parser.add_argument(
    "--gamma", type=float, default=0.95, help="gamma in Adam (default: 0.95)"
)

parser.add_argument(
    "--SGD", action="store_true", help="Use SGD as optimizer with momentum 0.9"
)


parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size used in training (default: 32)",
)

parser.add_argument(
    "--start_epoch",
    type=int,
    default=0,
    help="Epoch to start training with (default: 0)",
)

parser.add_argument(
    "--end_epoch",
    type=int,
    default=200,
    help="Epoch to end training (default: 200)",
)

parser.add_argument("-epoch_freq", "--save_latest_epoch_freq", type=int, default=100)

parser.add_argument("-logf", "--logfile", type=str, default="train_loss.txt")

parser.add_argument("--mean_range_dict", type=str)

parser.add_argument("--db_name", type=str, default="qppnet_db")
parser.add_argument("--db_user", type=str, default="qppnet_user")
parser.add_argument("--db_pass", type=str, default="qppnet_pass")

parser.add_argument(
    "--db_snapshot_path",
    type=str,
    default="database_snapshot.dill",
)


def save_opt(opt, logf):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        comment = ""
        default = parser.get_default(k)
        if v != default:
            comment = "\t[default: %s]" % str(default)
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    print(message)
    logf.write(message)
    logf.write("\n")


if __name__ == "__main__":
    opt = parser.parse_args()

    if opt.dataset == "POSTGRES":
        dataset = PostgresDataSet(opt)
        dim_dict = dataset.db_snapshot.dim_dict

    elif opt.dataset == "TerrierTPCH":
        raise NotImplementedError("Disabled.")
        dataset = TerrierTPCHDataSet(opt)
        with open("dataset/terrier_tpch_dataset/input_dim_dict.json", "r") as f:
            dim_dict = json.load(f)
    else:
        raise NotImplementedError("Disabled.")
        dataset = OLTPDataSet(opt)
        with open("./dataset/oltp_dataset/tpcc_dim_dict.json", "r") as f:
            dim_dict = json.load(f)

    print("dataset_size", dataset.datasize)
    torch.set_default_tensor_type(torch.FloatTensor)
    qpp = QPPNet(opt, dim_dict)

    total_iter = 0

    if opt.test_time:
        qpp.evaluate(dataset.all_dataset)
        print(
            "total_loss: {}; test_loss: {}; pred_err: {}; R(q): {}".format(
                qpp.last_total_loss, qpp.last_test_loss, qpp.last_pred_err, qpp.last_rq
            )
        )
    else:
        logf = open(opt.logfile, "w+")
        save_opt(opt, logf)
        # qpp.test_dataset = dataset.create_test_data(opt)
        qpp.test_dataset = dataset.test_dataset

        for epoch in range(opt.start_epoch, opt.end_epoch):
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

            samp_dicts = dataset.sample_data()
            total_iter += opt.batch_size

            qpp.set_input(samp_dicts)
            qpp.optimize_parameters(epoch)
            logf.write(
                "epoch: "
                + str(epoch)
                + "; iter_num: "
                + str(total_iter)
                + "; total_loss: {}; test_loss: {}; pred_err: {}; R(q): {}".format(
                    qpp.last_total_loss,
                    qpp.last_test_loss,
                    qpp.last_pred_err,
                    qpp.last_rq,
                )
            )

            # if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = qpp.get_current_losses()
            loss_str = "losses: "
            for op in losses:
                loss_str += str(op) + " [" + str(losses[op]) + "]; "

            if epoch % 50 == 0:
                print(
                    "epoch: "
                    + str(epoch)
                    + "; iter_num: "
                    + str(total_iter)
                    + "; total_loss: {}; test_loss: {}; pred_err: {}; R(q): {}".format(
                        qpp.last_total_loss,
                        qpp.last_test_loss,
                        qpp.last_pred_err,
                        qpp.last_rq,
                    )
                )
                print(loss_str)

            logf.write(loss_str + "\n")

            if (
                epoch + 1
            ) % opt.save_latest_epoch_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print(
                    "saving the latest model (epoch %d, total_iters %d)"
                    % (epoch + 1, total_iter)
                )
                qpp.save_units(epoch + 1)

        logf.close()
