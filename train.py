import tensorflow as tf
import numpy as np
import argparse
import yaml
import random

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

    # set timestamp to visualize results with plots
    ts_rain = 250
    ts_no_rain = 275

    # plot given meteorological data (to transform)
    plot_cube(train_x, ts_rain, train_t, config)
    # plot sample from given meteorological data
    plot_map(train_x[:,:,:,97], ts_rain, train_t, config['data']['output_path'] + 'input_sample.png')
    # plot given radar precipitation map (target reference)
    plot_map(train_y, ts_rain, train_t, config['data']['output_path'] + 'output_rain_tar.png')
    # plot no rain target reference
    plot_map(train_y, ts_no_rain, train_t, config['data']['output_path'] + 'output_no_rain_tar.png')

    # load model
    deconvNet = DeconvNet()

    # load hyperparameters
    n_epochs = config['training']['epochs']
    n_batches = int(train_x.shape[0] / config['training']['batches'])

    # initialize loss
    loss_function = tf.keras.losses.MeanSquaredLogarithmicError()
    # initialize optimizer
    optimizer = tf.keras.optimizers.SGD()

    # initialize lists for visualization
    train_losses = []
    test_losses = []

    # test once before we begin
    test_loss = test(deconvNet, test_x, test_y, loss_function, config)
    test_losses.append(test_loss)

    # check how model performs on train data once before we begin
    train_loss = test(deconvNet, train_x, train_y, loss_function, config)
    train_losses.append(train_loss)

    # train for n_epochs epochs
    for epoch in range(n_epochs):
        print("Epoch {} starting with loss {}".format(str(epoch),
              test_losses[-1]))
        
        epoch_loss = []

        # train (and check in with training)
        for batch in range(n_batches):
            start = batch * config['training']['batches']
            end = (batch + 1) * config['training']['batches']
            x_batch = train_x[start:end, :, :, :]
            y_batch = train_y[start:end, :, :]
            t_batch = train_t[start:end]

            (prediction, train_loss) = train_step(deconvNet, x_batch, y_batch, loss_function, optimizer)
            epoch_loss.append(train_loss)

            if batch % 100 == 0:
                sample_batch = random.randint(0, config['training']['batches'] - 1)
                plot_map(prediction, sample_batch, t_batch, config['data']['output_path'] + 'epoch' + str(epoch) + '_batch' + str(int(batch / 100)) + '_pred.png')
                plot_map(train_y, start + sample_batch, train_t, config['data']['output_path'] + 'epoch' + str(epoch) + '_batch' + str(int(batch / 100)) + '_tar.png')
        
        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss))

        # test, so we can track test loss
        test_loss = test(deconvNet, test_x, test_y, loss_function, config)
        test_losses.append(test_loss)

    # plot sample predictions for comparison to target
    pred_rain = deconvNet(np.expand_dims(train_x[ts_rain, :, :, :], axis=0))
    plot_map(np.expand_dims(pred_rain, axis=0), 0, train_t[ts_rain], config['data']['output_path'] + 'output_rain_pred.png')
    pred_no_rain = deconvNet(np.expand_dims(train_x[ts_no_rain, :, :, :], axis=0))
    plot_map(np.expand_dims(pred_no_rain, axis=0), 0, train_t[ts_no_rain], config['data']['output_path'] + 'output_no_rain_pred.png')

    # visualize loss for training and test data
    plot_train_process(train_losses, test_losses, config)

    # save model weights
    deconvNet.save_weights(config['data']['weight_path'] + 'deconv_weights')

    
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
        prediction = model(input, True)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return (prediction, loss)


def test(model, input, target, loss_function, config):
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
    """

    test_losses = []
    n_batches = int(input.shape[0] / config['training']['batches'])

    for batch in range(n_batches):
        x_batch = input[batch * config['training']['batches']:(batch + 1) * config['training']['batches'], :, :, :]
        y_batch = target[batch * config['training']['batches']:(batch + 1) * config['training']['batches'], :, :]
    
        prediction = model(x_batch, False)
        sample_test_loss = loss_function(y_batch, prediction)
        test_losses.append(sample_test_loss.numpy())

    test_loss = tf.reduce_mean(test_losses)

    return test_loss


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
