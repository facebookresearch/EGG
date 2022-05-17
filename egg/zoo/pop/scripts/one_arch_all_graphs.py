from egg.zoo.pop.scripts.acc_graphs import graph_collector, acc_graph


def one_architecture_all_exps(
    arch_name="inception",
    arch_as_sender=True,
    baselines=True,
    verbose=False,
    save_path="/shared/mateo/logs/",
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
            "vision_model_names_senders"
            if arch_as_sender
            else "vision_model_names_recvs",
            "vision_model_names_recvs"
            if arch_as_sender
            else "vision_model_names_senders",
            "recv_hidden_dim",
            "lr",
            "recv_output_dim",
            "n_epochs",
            "batch_size",
        ],
        values=[
            [[arch_name]],
            [
                ["resnet152"],
                ["vit"],
                ["inception"],
                ["vgg11"],
                ["vgg11", "vit", "resnet152", "inception"],
                ["vgg11", "inception", "vit", "resnet152"],
                # ["vit", "vit", "vit", "vit"]
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

    _labels = []
    _nothing_labeled = True
    # colour_iterator = iter(plt.cm.rainbow(np.linspace(0, 0.2, len(xs) + 1)))
    colours = []
    for l in labels:
        if l == f"{arch_name} --> {arch_name}":
            _l = l
            colours.append("r")
        elif l == f"{arch_name} --> diverse population":
            _l = l
            colours.append("limegreen")
        elif l == f"diverse population --> {arch_name}":
            _l = l
            colours.append("limegreen")
        elif _nothing_labeled:
            _l = (
                f"{arch_name} --> other architecture"
                if arch_as_sender
                else f"other architecture --> {arch_name}"
            )
            _nothing_labeled = False
            colours.append("purple")
        else:
            _l = None
            colours.append("purple")
        _labels.append(_l)
    labels = _labels

    if baselines:
        _xs, _ys, _labels = graph_collector(
            names=[
                "vision_model_names_senders",
                "vision_model_names_recvs",
                "additional_sender",
                "additional_receiver",
            ],
            values=[
                [["vgg11", "inception", "vit", "resnet152"]],
                [["vgg11", "inception", "vit", "resnet152"]],
                [None],
                [None],
            ],
            verbose=verbose,
        )
        xs += _xs
        ys += _ys
        labels += _labels
        colours += ["g"] * len(_xs)  # one specific colour

    # plot all aquired data
    acc_graph(
        xs,
        ys,
        labels,
        save_path,
        verbose,
        name=graph_name,
        title=arch_name if graph_title is None else graph_title,
        legend_title="sender --> receiver",
        colours=colours,
    )

if __name__ == '__main__':
    one_architecture_all_exps()