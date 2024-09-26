import pandas as pd

# import torch
# import scipy


def interaction_to_dataframe(interaction):
    """
    Function to turn the Interaction file into a pandas DataFrame which is covered with syntaxic sugar so easy to use
    """
    df = pd.DataFrame()
    # skipped as is empty in the emecom_pop case
    # df["sender_input"] = interaction.sender_input.numpy()
    # df["receiver_input"] = interaction.receiver_input
    df["labels"] = interaction.labels

    for key in interaction.aux_input:
        if (
            key == "receiver_message_embedding"
        ):  # in continuous format message and receiver embedding are the same
            for dim, value in enumerate(interaction.message.T):
                df[f"dim_{dim}"] = value
        elif "vision" in key:
            continue
        else:
            df[key] = interaction.aux_input[key]
    df["receiver_output"] = interaction.receiver_output.argmax(dim=1)

    return df


def name_to_idx(name):
    """
    Function to convert a name to the vision-module index
    """
    names = ["vgg11", "vit", "resnet152", "inception", "swin", "dino", "virtex"]
    assert name in names, f"{name} is not a valid vision-module name"
    return names.index(name)


def extract_name(name):
    """
    Function to extract the name of the vision-module from the interaction file name
    """
    return [i.replace("]", "").replace("'", "") for i in name.split("[")[1:]]

if __name__ == "__main__":
    import torch
    interaction=torch.load("/home/mmahaut/projects/exps/tmlr/v64_com_sender_rep/imagenet_valcontinuous64None['vgg11']['vit']")
    df2=pd.DataFrame(interaction.aux_input["vision_module"])
    df1=pd.DataFrame(interaction.aux_input["receiver_message_embedding"])
    # df3=pd.DataFrame(interaction.aux_input["path"])
    print(interaction.aux_input.keys())
    input(df1.head())
    input((df2.head()))
    # input((df3.head()))
    df1.to_csv("/home/mmahaut/temp_interactions/receiver_message_embedding.csv")
    df2.to_csv("/home/mmahaut/temp_interactions/vision_module.csv")
    # df3.to_csv("/home/mmahaut/temp_interactions/path.csv")

