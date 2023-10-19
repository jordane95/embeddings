
# plot the training loss during training at different steps

import json
import matplotlib.pyplot as plt
import numpy as np

# read loss trainer state from file
def read_loss_state(file_path):
    with open(file_path, 'r') as f:
        loss_state = json.load(f)
    losses = loss_state["log_history"] # List[Dict]
    return losses

# plot the training loss
def plot_loss(losses, save_path):
    # get the loss values
    loss_values = []
    steps = []
    for loss in losses:
        loss_values.append(loss["loss"])
        steps.append(loss["step"])
    # plot the loss values
    plt.plot(steps, loss_values)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.savefig(save_path)
    plt.close()


# plot and compare the loss values of different models
def plot_loss_compare(model_losses, save_path):
    # model_losses: Dict[model_name, List[loss_values]]
    # plot the loss values
    # set to plot to be embeded in an academic paper
    # set the style
    plt.style.use('seaborn-whitegrid')
    # double the font scale
    plt.rcParams.update({'font.size': 24})
    plt.rcParams.update({'figure.figsize': [6.4, 4.8]})
    # avoid y label being truncated
    plt.rcParams.update({'figure.autolayout': True})

    # set max y value
    plt.ylim(0, 10.0)
    
    for model_name, loss_values in model_losses.items():
        steps = []
        losses = []
        for loss in loss_values:
            steps.append(loss["step"])
            losses.append(loss["loss"])
        # plot the loss values and smmoth the curve
        plt.plot(steps, losses, label=model_name)
    # set the x and y axis
    plt.xlabel("steps (k)")
    plt.ylabel("loss")
    # resalce the x axis with 1k as scale
    plt.xticks(np.arange(0, 10001, 1000), np.arange(0, 11, 1))
    plt.legend()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # read loss state
    file_pathes = {
        "small": "ckpt/pt_graphcodebert_so_dec_csn_2a3096bs128msl10k2e-4/checkpoint-10000/trainer_state.json",
        # "base": "base_trainer_state.json",
        # "large": "large_trainer_state.json",
    }
    losses = {
        model: read_loss_state(file_pathes[model])
        for model in file_pathes
    }
    # plot the loss
    save_path = "loss"
    plot_loss_compare(losses, save_path)

