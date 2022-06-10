from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import tensorflow as tf
import numpy as np
import argparse
import yaml

from models.deconvNet import DeconvNet
from tools.load_data import load_data
from tools.plot_data import plot_cube, plot_map, plot_train_process


def train_DeconvNet(config):
    """
    Train DeconvNet
        Args:
            config <dict> config file containing directory paths and 
                          hyperparameters
    """

    # load training data
    train_x, train_y, train_c, train_t = load_data(config, 'train')
    # load test data
    test_x, test_y, test_c, test_t = load_data(config, 'test')

    # set timestamp to check results with plots
    ts = 250

    # plot given meteorological data (to transform)
    plot_cube(train_x, ts, train_t, config)
    # plot given radar precipitation map (result reference)
    plot_map(train_y, ts, train_t, config['data']['output_path'] + 'output_reference.png')

    # load model
    deconvNet = DeconvNet()

    #TODO add more if necessary, e.g. learning rate
    # load hyperparameters
    n_epochs = config['training']['epochs']

    #TODO do via config, use different loss (pred values can be negative)
    # initialize loss
    loss_function = tf.keras.losses.MeanSquaredError()
    # initialize the optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    # initialize lists for visualization
    train_losses = []
    test_losses = []
    test_accuracies = []

    # testing once before we begin
    test_loss, test_accuracy = test(deconvNet, test_x, test_y, loss_function)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # check how model performs on train data once before we begin
    train_loss, _ = test(deconvNet, train_x, train_y, loss_function)
    train_losses.append(train_loss)

    #TODO use batches

    # train for n_epochs epochs, starting with epoch 0
    for epoch in range(n_epochs):
        print("Epoch: {} starting with accuracy {}".format(str(epoch),
              test_accuracies[-1]))

        # training (and checking in with training)
        epoch_loss = []
        for observation in range(train_x.shape[0]):
            (prediction, train_loss) = train_step(deconvNet, train_x[observation, :, :, :], train_y[observation, :, :], loss_function, optimizer)
            epoch_loss.append(train_loss)

            if observation % 5000 == 0:
                plot_map(np.expand_dims(prediction, axis=0), 0, train_t[observation], config['data']['output_path'] + 'epoch_' + str(epoch) + '_' + str(int(observation / 5000)) + '.png')
        
        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss))

        # testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(deconvNet, test_x, test_y, loss_function)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    plot_map(train_y, ts, train_t, config['data']['output_path'] + 'output_final.png')

    # visualize accuracy and loss for training and test data
    plot_train_process(train_losses, test_losses, test_accuracies)

    
def train_step(model, input, target, loss_function, optimizer):
    """
    Train model
        Args:
            model: <tf.keras.Model> model to train
            input: <np.ndarray> training data (model input)
            target: <np.ndarray> target for model training
            loss_function: <tf.keras.losses> loss function (calculating loss 
                            given model prediction and target)
            optimzer: <tf.keras.optimizers> optimizer (improving model)
        Returns:
            prediction: <tf.Tensor> model output
            loss: <tf.Tensor> loss value
    """

    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return (prediction, loss)


def test(model, input, target, loss_function):
    """
    Test model
        Args:
            model: <tf.keras.Model> model to test
            input: <np.ndarray> test data (model input)
            target: <np.ndarray> target for model test
            loss_function: <tf.keras.losses> loss function (calculating loss 
                            given model prediction and target)
        Returns:
            test_loss: <tf.Tensor> loss calculated when testing model
            test_accuracy: <tf.Tensor> accuracy calculated when testing model
    """

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for observation in range(input.shape[0]):
        prediction = model(input[observation, :, :, :])
        sample_test_loss = loss_function(target[observation, :, :], prediction)
        sample_test_accuracy =  np.argmax(target[observation, :, :], axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--model",
        nargs="?",
        type=str,
        default="DeconvNet",
        help="Model to train"
    )
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./configs/deconvNet.yml",
        help="Model to train"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    if args.model == "DeconvNet":
        train_DeconvNet(config)
    else:
        print("No valid model found")
