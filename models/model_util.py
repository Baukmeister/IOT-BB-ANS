from random import sample

from matplotlib import pyplot as plt

# CONFIG
max_points = 100

def plot_prediction(prediction_tensors, target_tensors, batch_idx, loss):


    # select a random subset of the target and predictions to not overcrowd the plot
    predictions = sample(
        [prediction.cpu().detach().numpy() for prediction in prediction_tensors],
        min(max_points, prediction_tensors.shape[0])
    )
    targets = sample(
        [target.cpu().detach().numpy() for target in target_tensors],
        min(max_points, target_tensors.shape[0])
    )

    pred_x = [[pred[i] for i in range(0, len(pred), 3)] for pred in predictions]
    pred_y = [[pred[i] for i in range(1, len(pred), 3)] for pred in predictions]
    pred_z = [[pred[i] for i in range(2, len(pred), 3)] for pred in predictions]

    target_x = [[target[i] for i in range(0, len(target), 3)] for target in targets]
    target_y = [[target[i] for i in range(1, len(target), 3)] for target in targets]
    target_z = [[target[i] for i in range(2, len(target), 3)] for target in targets]

    ### PLOT ONLY GROUND TRUTH
    fig = plt.figure(figsize=(4, 4))

    ax = fig.add_subplot(projection='3d')

    ax.scatter(
        [item for sublist in target_x for item in sublist],
        [item for sublist in target_y for item in sublist],
        [item for sublist in target_z for item in sublist]
        , c="red")

    plt.title(f"Batch: {batch_idx} - Ground Truth")
    plt.show()


    ### PLOT GROUND TRUTH + PREDICTIONS
    fig = plt.figure(figsize=(4, 4))

    ax = fig.add_subplot(projection='3d')

    # plot the points
    pred_ax = ax.scatter(
        [item for sublist in pred_x for item in sublist],
        [item for sublist in pred_y for item in sublist],
        [item for sublist in pred_z for item in sublist]
        , c="blue")

    target_ax = ax.scatter(
        [item for sublist in target_x for item in sublist],
        [item for sublist in target_y for item in sublist],
        [item for sublist in target_z for item in sublist]
        , c="red"
        , alpha=0.3)

    plt.legend([pred_ax, target_ax], ["Predictions", "Targets"])
    plt.title(f"Batch: {batch_idx} - Training loss: { round(loss.item(), 5)}")
    plt.show()




