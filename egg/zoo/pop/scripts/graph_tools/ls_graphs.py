from egg.zoo.pop.scripts.graph_tools.acc_graphs import graph_collector, acc_graph


def ls_graph(
    save_path="/shared/mateo/logs/",
    graph_name="arch_graph",
    graph_title="no_inception",
    verbose=False,
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
            # "additional_sender",
            # "additional_receiver",
        ],
        values=[
            [["vgg11", "vit", "inception", "resnet152"]],
            [["vgg11", "vit", "inception", "resnet152"]],
        ],
        verbose=verbose,
        label_names=["additional_sender", "additional_receiver"],
    )
    colours = ["g"] * len(xs)  # one specific colour
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
        legend_title="add_sender - add_receiver",
        # colours=colours,
    )
