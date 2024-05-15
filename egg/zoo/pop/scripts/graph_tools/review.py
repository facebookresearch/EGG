from matplotlib import pyplot as plt
import seaborn as sns
import glob
import json
from egg.zoo.pop.scripts.sweeper import sweep_params, get_opts
from pathlib import Path
# import retex as rt

stats = {}
for fp in glob.glob(
    "/homedtcl/mmahaut/projects/experiments/discrete_repeat2/*/wandb/latest-run/files/wandb-metadata.json"
):
    with open(fp) as f:
        a = json.load(f)["args"]
        for el in a:
            if "vocab_size" in el:
                if int(el[13:]) not in stats.keys():
                    stats[int(el[13:])] = 1
                else:
                    stats[int(el[13:])] += 1
                    if int(el[13:]) == 8192:
                        print(fp)
print(stats)

data = {}
data["vocab_size"] = []
data["run"] = []
data["type"] = []
data["grad"] = []
base_path = "/home/mmahaut/projects/experiments/"
prefix = "rev_vir_16"
full_path = Path(base_path) / prefix
for fp in full_path.glob(f"*.out"):
    job_number = str(fp).split( "_")[-1].split(".")[0]
    with open(
        Path(full_path) / job_number / "wandb" / "latest-run" / "files" / "wandb-metadata.json"
    ) as f:
        a = json.load(f)["args"]
        for el in a:
            if "vocab_size" in el:
                vocab_size = int(el[13:])
                print(vocab_size)
                break
    if vocab_size == 16:
        with open(fp) as f:
            a = f.readlines()
            for i, line in enumerate(a):
                if "norm" in line and "game" not in line:
                    # _dat = json.loads(line)
                    # add all elements of line to data
                    # for k in _dat.keys():
                    #     if k not in data.keys():
                    #         data[k] = []
                    #     data[k].append(_dat[k])
                    if line.split(" ")[0] not in data.keys():
                        data[line.split(" ")[0]] = []
                    data["grad"].append(float(line.split(" ")[-1]))
                    data["run"].append(i)
                    data["type"].append(line.split(" ")[0])
                    if i >= 5000 and float(line.split(" ")[-1]) >= 0.3:
                        print(i, line)
                        break
                    # data["vocab_size"].append(vocab_size)
            # else:
# make plots for each key in data
# for i, k in enumerate(data.keys()):
#     if k == "vocab_size":
#         continue
#     # sns.lineplot(x=data["epoch"], y=data[k], hue=data["vocab_size"])
#     sns.lineplot(
#         x=data["run"][: len(data[k])],
#         y=data[k],
#         # hue=data["vocab_size"][i * len(data[k]) : (i + 1) * len(data[k])],
#         palette="colorblind",
#     )
#     plt.title(k)
#     plt.savefig(f"{k}.png")
#     plt.close()
# print(data["type"])
sns.lineplot(x=data["run"], y=data["grad"], hue=data["type"], palette="colorblind")

# add one in
params = {
    "n_epochs": [30],
    "dataset_dir": ["/datasets/COLT/ILSVRC2012/ILSVRC2012_img_val/"],
    "dataset_name": ["imagenet_val"],
    "image_size": [384],
}
models = ["vgg11", "vit", "resnet152", "inception", "dino", "swin", "virtex"]
for f in glob.glob(
    "/homedtcl/mmahaut/projects/experiments/rev_16_partial_pop/*/final.tar"
):
    params["base_checkpoint_path"] = [f]
    args = json.load(open(f[:-9] + "wandb/latest-run/files/wandb-metadata.json"))[
        "args"
    ]
    mod = []
    for p in args:
        if "vision_model_names_senders" in p:
            print(p)
            [mod.append(x) for x in models if x not in mod and x not in eval(p[29:])]
            print(mod)
        if "vision_model_names_recvs" in p:
            [mod.append(x) for x in models if x not in mod and x not in eval(p[27:])]
    print(mod, models)
    for m in mod:
        # launch sweep
        params["additional_sender"] = [f"['{m}']"]
        with open("temp.json", "w") as out:
            json.dump(params, out)
        print(params)
        opts = [
            "--params_path=temp.json",
            "--job_name=rev_16_add_one_pop",
            "--game=/homedtcl/mmahaut/projects/EGG/egg/zoo/pop/train.py",
        ]
        sweep_params(get_opts(opts))
        params.pop("additional_sender")
        params["additional_receiver"] = [f"['{m}']"]
        with open("temp.json", "w") as out:
            json.dump(params, out)
        # sweep_params(get_opts(opts))
        params.pop("additional_receiver")

# leave on out
params = {
    "n_epochs": [25],
    "fp16": [False],
    "no_cuda": [False],
    "batch_size": [64],
    "lr": [0.0001],
    "continuous_com": [True],
    "com_channel": ["continuous"],
    "non_linearity": ["sigmoid"],
    "force_gumbel": [False],
    "vocab_size": [16],
    "random_seed": [111],
    "recv_hidden_dim": [2048],
    "dataset_dir": ["/datasets/COLT/ILSVRC2012/ILSVRC2012_img_val/"],
    "dataset_name": ["imagenet_val"],
    "gs_temperature": [5.0],
    "image_size": [384],
}
models = ["vgg11", "vit", "resnet152", "inception", "dino", "swin", "virtex"]
# make all tables of models removing one
for i in range(len(models)):
    partial_models = models[:i] + models[i + 1 :]
    params["vision_model_names_senders"] = [f"{partial_models}"]
    params["vision_model_names_recvs"] = [f"{partial_models}"]
    with open("temp.json", "w") as out:
        json.dump(params, out)
    print(params)
    opts = [
        "--params_path=temp.json",
        "--job_name=rev_16_partial_pop",
        "--game=/homedtcl/mmahaut/projects/EGG/egg/zoo/pop/train.py",
    ]
    sweep_params(get_opts(opts))
    params.pop("vision_model_names_senders")
    params.pop("vision_model_names_recvs")

# import json
# import glob
# import seaborn as sns
# import matplotlib.pyplot as plt
# #smooth loss
# data={}
# #collect loss data for epochs 1 to 30 for all files
# #plot snd save
# for f in glob.glob("/gpfs/home/mmahaut/projects/experiments/rev_vir_16/*.out"):
#     print(f)
#     with open(f) as f:
#         a = f.readlines()
#         for i, line in enumerate(a):
#             if i == 0:
#                 continue
#             if "loss" in line:
#                 args=eval(line)
#                 if "loss" in args.keys():
#                     if "loss" not in data.keys():
#                         data["loss"]=[]
#                     print(args["loss"])
#                     data["loss"].append(args["loss"])
#                     if "epoch" not in data.keys():
#                         data["epoch"]=[]
#                     data["epoch"].append(args["epoch"])
# # plot loss
# sns.lineplot(x=data["epoch"], y=data["loss"])
# plt.title("loss")
# tight_layout()
# x axis is Learning step
plt.xlabel("Learning step")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("grad_norm_cont.png")
plt.close()


