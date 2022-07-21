import pandas as pd
import torch
import scipy

def interaction_to_dataframe(interaction):
    """
    Function to turn the Interaction file into a pandas DataFrame which is covered with syntaxic sugar so easy to use
    """
    df = pd.DataFrame()
    # skipped as is empty in the emecom_pop case
    # df["sender_input"] = interaction.sender_input
    # df["receiver_input"] = interaction.receiver_input
    df["labels"] = interaction.labels
    for key in interaction.aux_input:
        if key == "receiver_message_embedding": # in continuous format message and receiver embedding are the same
            for dim, value in enumerate(interaction.message.T):
                df[f"dim_{dim}"] = value
        else:
            df[key] = interaction.aux_input[key]
    df["receiver_output"] = [i.argmax().item() for i in interaction.receiver_output]

    return df




