import matplotlib.pyplot as plt
import json
import glob
import os


def text_to_data(file_path, mode="train"):  # Mat going through console
    with open(file_path) as f:
        x = []
        y = []
        lines = f.readlines()
        for line in lines:
            if "{" in line:
                _dict = json.loads(line)
                if _dict["mode"] == "train":
                    x.append(_dict["epoch"])
                    y.append(_dict["acc"])
        return x, y


def extract_metadata(path, verbose=False):
    with open(path) as file:
        meta = json.load(file)
        result = ""
        for arg in meta["args"]:
            if "--vision_model_names_recvs" in arg:
                result += arg[29 : len(arg) - 2] + " "
            if "--vision_model_names_senders" in arg:
                result += arg[31 : len(arg) - 2]
        if verbose:
            print(result)
        return result


def extract_meta_from_output():

    pass


def get_log_files(wandb_path):
    return [file for file in glob.glob(wandb_path + "/run*")]


def make_acc_graph(wandb_path="/mnt/efs/fs1/EGG/wandb", verbose=False):
    for file_path in get_log_files(wandb_path):
        print(f"extracting data from {file_path}")
        x, y = text_to_data(os.path.join(file_path, "files/output.log"), mode="train")
        plt.plot(
            x,
            y,
            label=extract_metadata(
                os.path.join(file_path, "files/wandb-metadata.json"), verbose=verbose
            ),
        )
        plt.legend()
        # plt.title("r={}")
    plt.savefig(os.path.join(wandb_path, "acc_graph.png"))


def nest_acc_graph():
    pass


make_acc_graph(verbose=True)
# print(extract_metadata("D:/alpha/EGG/egg/zoo/pop/test.json"))
