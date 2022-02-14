from unicodedata import name
import matplotlib.pyplot as plt
import json
import glob
import os

# 1st Harvest data, either from wnb or from nest output. Then make a graph out of it

##
## Data Harvest -- general
##


def metadata_opener(file, data_type: str):
    """
    data_type : str in {"wandb", "nest"}
    Mat : case match in python 3.10 will cover all of this in syntaxic sugar
    """
    if data_type == "wandb":
        meta = json.load(file)
        return meta["args"]

    if data_type == "nest":
        # in nest, metadata are written as comments on the first line of the .out file
        # TODO: only parameters used in the sweep json file are available here.
        # All other parameters will be set to default values but will not appear here
        # A future version of this opener should take into account the Namespace object on the following line
        lines = file.readlines()
        assert (
            lines[0][0] == "#"
        )  # Making sure we're in the right place on the right line
        params = eval(lines[0][2:])  # Mat : injection liability
        return params

    else:
        raise KeyError(
            f"{data_type} is not a valid type for data_type in metadata_opener"
        )


def extract_model_names(args, verbose=False):
    result = ""
    for arg in args:
        if "--vision_model_names_senders" in arg:
            result = arg[31 : len(arg) - 2] + result
        if "--vision_model_names_recvs" in arg:
            result = result + arg[29 : len(arg) - 2] + " "
    if verbose:
        print(result)
    return result


def check_constraints(path, names=[], values=[]):
    """
    Default parameters result in automatic pass
    """
    respects_constraints = True
    assert len(names) == len(values)
    # Maybe, only the file opener needs to be different
    with open(path) as f:
        params = metadata_opener(f)
        for i in range(len(values)):
            if not extract_param(names[i], params) in values[i]:
                respects_constraints = False

    return respects_constraints


def extract_param(param_name, params, verbose=False):
    for param in params:
        if param_name in param:
            result = param[len(param_name) + 2 : len(param)]
            if verbose:
                print(result)
            return result
    raise KeyError(
        f"{param} was not found amongs parameters"
    )  # Mat : is it a Keyerror tho ?


def text_to_acc(file_path, mode="train"):  # Mat : going through console
    with open(file_path) as f:
        x = []
        y = []
        lines = f.readlines()
        for line in lines:
            if "{" in line:
                _dict = json.loads(line)
                if _dict["mode"] == mode:
                    x.append(_dict["epoch"])
                    y.append(_dict["acc"])
        return x, y


##
## Data harvest wandb
##


def extract_meta_from_wnb(path, verbose=False):
    # rendered obsolete by metadata_opener and extract_param
    with open(path) as file:
        meta = json.load(file)
        return extract_model_names(meta["args"], verbose=verbose)


def make_wnb_graph(wandb_path="/mnt/efs/fs1/EGG/wandb", verbose=False):
    xs = []
    ys = []
    labels = []
    for file_path in glob.glob(wandb_path + "/run*"):
        if verbose:
            print(f"extracting data from {file_path}")
        x, y = text_to_acc(os.path.join(file_path, "files/output.log"), mode="train")
        xs.append(x)
        ys.append(y)
        labels.append(
            extract_meta_from_wnb(
                os.path.join(file_path, "files/wandb-metadata.json"), verbose=verbose
            )
        )
    acc_graph(xs, ys, labels, wandb_path)


##
## Data harvest nest
##
def extract_meta_from_nest_out(file_path, verbose=False):
    with open(file_path) as f:
        lines = f.readlines()
        # Mat : from the nest output we only need the first line
        params = eval(lines[0][2:])  # Mat : injection liability
        return extract_model_names(params, verbose=verbose)


def nest_acc_graph(nest_path="~/nest_local/", names=[], values=[], verbose=False):
    """
    restrict nest_path to a more specific experiment to only search there
    TODO : three file access seems too much, do it in one
    """
    xs = []
    ys = []
    labels = []
    for file_path in glob.glob(nest_path + "*.out"):
        # restrict to specific parameters
        if check_constraints(file_path, names, values):
            labels.append(extract_meta_from_nest_out(file_path))
            x, y = text_to_acc(file_path)
            xs.append(x)
            ys.append(y)
    acc_graph(xs, ys, labels, nest_path, verbose)


## Graph making
def acc_graph(xs, ys, labels, save_path="~/graphs", verbose=False):
    # maybe to add a better file naming system, preventing overwrite
    assert len(xs) == len(ys) == len(labels)

    for i in range(len(xs)):
        if verbose:
            print(f"adding {labels[i]} to graph")
        plt.plot(
            xs[i],
            ys[i],
            label=labels[i],
        )
        plt.legend()
        # plt.title("r={}")
    with open(os.path.join(save_path, "acc_graph.png")) as f:
        plt.savefig(f)


## Execution
if __name__ == "__main__":
    nest_acc_graph(
        names=["lr", "vocab_size", "batch_size"],
        values=[[0.5, 1, 2.4], [256, 512, 1024], [8, 16]],
        # values=[[0.5, 1, 2.4], [256, 512, 1024], [8, 16]],
        verbose=True,
    )
    # print(extract_metadata("D:/alpha/EGG/egg/zoo/pop/test.json"))
    