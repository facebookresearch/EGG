import matplotlib.pyplot as plt
import json
import glob
import os
import numpy as np

# 1st Harvest data, either from wnb or from nest output. Then make a graph out of it

##
## Data Harvest -- general
##


def metadata_opener(file, data_type: str, verbose=False):
    """
    data_type : str in {"wandb", "nest"}
    Mat : case match in python 3.10 will cover all of this in syntaxic sugar
    """
    if data_type == "wandb":  # TODO : move to the yaml file in the near future
        meta = json.load(file)
        return meta["args"]

    if data_type == "nest":
        # in nest, metadata are written as comments on the first line of the .out file
        # TODO: only parameters used in the sweep json file are available here.
        # All other parameters will be set to default values but will not appear here
        # A future version of this opener should take into account the Namespace object on the following line
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i][0] == "#":
                params = eval(lines[i][12:])  # Mat : injection liability
                return params
        if verbose:
            print("failed to find metadata in file")
        return []

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


def check_constraints(path, names=[], values=[], verbose=False):
    """
    Default parameters result in automatic pass
    """
    respects_constraints = True
    assert len(names) == len(values)
    # Check and notify for empty files
    if os.stat(path).st_size == 0:
        if verbose:
            print(f"{path} is empty")
        respects_constraints = False
    else:
        with open(path) as f:
            params = metadata_opener(
                f,
                "wandb" if path[len(path) - 4 : len(path)] == "json" else "nest",
                verbose=verbose,
            )
            for i in range(len(values)):
                _ep = extract_param(names[i], params, verbose=False)
                if _ep != "":
                    _ep = eval(_ep)
                if verbose:
                    print(f"{_ep} in {values[i]} --> {_ep in values[i]} ")
                if not _ep in values[i]:
                    respects_constraints = False
    if verbose:
        print(f"constraints respected : {respects_constraints}")
    return respects_constraints


def extract_param(param_name, params, verbose=False):
    for param in params:
        if param_name in param:
            result = param[len(param_name) + 3 : len(param)]
            if verbose:
                print(result)
            return result
    # raise KeyError(
    if verbose:
        print(
            f"{param_name} was not found amongst parameters"
        )  # Mat : is it a Keyerror tho ?
    return ""


def text_to_acc(file, mode="test", verbose=False):  # Mat : going through console
    x = []
    y = []
    lines = file.readlines()
    for line in lines:
        if "{" in line:
            _dict = json.loads(line)
            if _dict["mode"] == mode:
                if verbose:
                    print(_dict)
                x.append(_dict["epoch"])
                y.append(_dict["acc"])
    if verbose and x == []:
        print("file opened but no data was available")
    return x, y


##
## Data harvest wandb
##


def extract_meta_from_wnb(path, verbose=False):
    # rendered obsolete by metadata_opener and extract_param
    with open(path) as file:
        meta = json.load(file)
        return extract_model_names(meta["args"], verbose=verbose)


def nest_graph(
    path="/shared/mateo/logs/",
    save_path="/shared/mateo/logs",
    names=[],
    values=[],
    label_names=["vision_model_names_senders", "vision_model_names_recvs"],
    verbose=False,
    graph_name="nest",
    graph_title=None,
    mode="test",
    epoch_limit=None,
):
    """
    restrict nest_path to a more specific experiment to only search there
    TODO : three file access seems too much, do it in one
    """
    xs = []
    ys = []
    labels = []
    # find all folders in log path
    files = glob.glob(path + "/*/*.out")  # TODO : average accross multiple seeds
    if verbose and files == []:
        print(f"no files were found in path {path}")
    for file_path in files:
        # prevent experiments that crashed without generating files to show (as well as any empty folder)
        if os.path.exists(file_path):
            # restrict to specific parameters
            if check_constraints(
                file_path,
                names,
                values,
                verbose,
            ):
                with open(file_path) as f:
                    params = metadata_opener(
                        f, "nest", verbose=verbose
                    )  # TODO : make one file call for the whole function
                    label = ""
                    for _ln in label_names:
                        label += str(extract_param(_ln, params, verbose=False))
                    # data is added to those needing to be plotted when it respects the constraints
                    labels.append(label)
                x, y = text_to_acc(file_path, verbose=verbose, mode=mode)
                xs.append(x if epoch_limit is None else x[:epoch_limit])
                ys.append(y if epoch_limit is None else y[:epoch_limit])

    # plot all aquired data
    acc_graph(
        xs,
        ys,
        labels,
        save_path,
        verbose,
        name=graph_name,
        title=f"{graph_name[:-4]}_{mode}_acc" if graph_title is None else graph_title,
    )


##
## Data harvest nest
##


def extract_meta_from_nest_out(file_path, verbose=False):
    with open(file_path) as f:
        lines = f.readlines()
        # Mat : from the nest output we only need the first line
        params = eval(lines[0][2:])  # Mat : injection liability
        return extract_model_names(params, verbose=verbose)


## Graph making
def acc_graph(
    xs,
    ys,
    labels,
    save_path="~/graphs",
    verbose=False,
    name="graph.png",
    title="train_acc",
    legend_title=None,
    colours=None,
    linestyles=None,
):
    # TODO : add a better file naming system, preventing overwrite
    assert len(xs) == len(ys) == len(labels)
    for i in range(len(xs)):
        if verbose:
            print(f"adding {labels[i]} to graph")
        plt.plot(
            xs[i],
            ys[i],
            label=labels[i],
            c=colours[i] if colours is not None else None,
            linestyle=linestyles[i] if linestyles is not None else None,
        )
        plt.legend(title=legend_title)
        plt.title(title)
    plt.savefig(os.path.join(save_path, name))
    plt.clf()


def graph_collector(
    path="/shared/mateo/logs/",
    names=[],
    values=[],
    verbose=False,
    mode="test",
    epoch_limit=None,
    get_wandb=False,
    label_names=["vision_model_names_senders", "vision_model_names_recvs"],
):
    """
    redoing this for a specific graph format. This has been decommented, but todos have not been dealt with
    !! if more than one architecture is used, the label will be set to 'all architectures'
    """
    # hard-coded parameter
    # "['vgg11', 'vit', 'resnet152', 'inception']"
    # data collectors
    xs = []
    ys = []
    labels = []

    # get all available files
    files = glob.glob(path + "/*/*.out")
    if get_wandb:
        files += glob.glob(path + "/*/*/*/files/output.log")
    if verbose and files == []:
        print(f"no files were found in path {path}")

    # select desired files
    for file_path in files:
        is_nest_data = file_path[-4:] == ".out"
        if os.path.exists(file_path):
            if check_constraints(
                file_path if is_nest_data else file_path[:-10] + "wandb-metadata.json",
                names,
                values,
                verbose,
            ):
                label = ""
                # collect data
                with open(
                    file_path
                    if file_path[-4:] == ".out"
                    else file_path[:-10] + "wandb-metadata.json"
                ) as f:
                    if verbose:
                        print(file_path)
                    params = metadata_opener(
                        f,
                        "nest" if is_nest_data else "wandb",
                        verbose=verbose,
                    )

                    # generate labels
                    if label_names is not None and label_names != []:
                        for _l in label_names:
                            label += extract_param(_l, params, verbose=False) + "-"

                        # _sender_label = (
                        #     "diverse population"
                        #     if len(_sender_label) > 1
                        #     else _sender_label[0]
                        # )
                        # _recv_label = (
                        #     "diverse population"
                        #     if len(_recv_label) > 1
                        #     else _recv_label[0]
                        # )

                        # label = f"{_sender_label} --> {_recv_label}"
                    else:
                        label = None

                    x, y = text_to_acc(file_path, verbose=verbose, mode=mode)
                if len(x) > 0:
                    xs.append(x if epoch_limit is None else x[:epoch_limit])
                    ys.append(y if epoch_limit is None else y[:epoch_limit])
                    labels.append(
                        label[:-1]
                    )  # removing the extra dash added by the final loop

    return xs, ys, labels
