from egg.zoo.pop.scripts.graph_tools.acc_graphs import graph_collector, acc_graph


def all_one_on_one(
    baselines=True,
    verbose=False,
    save_path='/homedtcl/mmahaut/projects/experiments/',
    graph_name="arch_graph",
    graph_title=None,
):
    """
    params
    ------
    arch_name : string, {'vgg11', 'vit', 'inception', 'resnet152'}
        which architecture's data will be used in the graph
    baselines : bool
        whether baselines are also to be plotted (for now, only the full population baseline is available)
    """

    # xmin, xmax, ymin, ymax = axis()
    # xmin, xmax, ymin, ymax = axis([xmin, xmax, ymin, ymax])

    # sender graph
    xs, ys, labels = graph_collector(
        names=[
            "vision_model_names_senders",
            "vision_model_names_recvs",
            "recv_hidden_dim",
            "lr",
            "recv_output_dim",
            "n_epochs",
            "batch_size",
        ],
        values=[
            [
                ["resnet152"],
                ["vit"],
                ["inception"],
                ["vgg11"],
            ],
            [
                ["resnet152"],
                ["vit"],
                ["inception"],
                ["vgg11"],
            ],
            [2048],
            [0.0001],
            [512],
            [25],
            [64],
        ],
        verbose=verbose,
    )

    # correcting and simplifying labels on a case by case basis
    # adding slightly different colours for the different elements

    colours = []
    for l in labels:
        # # receiver
        # if "--> vgg1" in l:
        #     linestyles.append("-.")
        # elif "--> vit" in l:
        #     linestyles.append("--")
        # elif "--> resnet152" in l:
        #     linestyles.append("-")
        # elif "--> inception" in l:
        #     linestyles.append(":")
        # sender
        if "vgg11 -->" in l:
            colours.append("r")
        elif "vit -->" in l:
            colours.append("limegreen")
        elif "resnet152 -->" in l:
            colours.append("b")
        elif "inception -->" in l:
            colours.append("purple")

    if baselines:
        _xs, _ys, _labels = graph_collector(
            names=[
                "vision_model_names_senders",
                "vision_model_names_recvs",
                "additional_sender",
                "additional_receiver",
            ],
            values=[
                [["vgg11", "vit", "resnet152", "inception"]],
                [["vgg11", "vit", "resnet152", "inception"]],
                [],
                [],
            ],
            verbose=verbose,
        )
        xs += _xs
        ys += _ys
        labels += _labels
        colours += ["g"] * len(_xs)  # one specific colour
        # linestyles += ["-"] * len(_xs)

    # plot all aquired data
    acc_graph(
        xs,
        ys,
        labels,
        save_path,
        verbose,
        name=graph_name,
        title=graph_title,
        legend_title="sender --> receiver",
        colours=colours,
    )
