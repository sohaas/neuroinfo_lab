import tensorflow as tf
import numpy as np
import argparse
import yaml

from models.deconvNet import DeconvNet
from tools.load_data import load_data
from tools.plot_data import plot_cube, plot_map


#TODO what to do with cosmo prediction ?
#TODO what to do with timepoints (only train on data from one year) ?

def train_Model(config):

    """
    Load data
    """
    train_data = load_data(config, 'train')
    train_ds = train_data[0]

    #test_data = load_data(config, 'test')
    #test_ds = test_data[0]

    # random timestamp to check results with plots
    ts = 250

    plot_cube(train_ds, ts, train_data[3], config)
    plot_map(train_data[1], ts, train_data[3], config)

    """
    # Optimize latent vector
    optimizer_class = get_optimizer(config, 'model')
    optimizer_params = {
        k: v for k, v in config['optimizer']['model'].items() if k != 'name'}
    optimizer = optimizer_class(**optimizer_params)

    # load model
    deconvNet = DeconvNet()

    epoch = 0
    model_losses = []
    train_M_losses = []

    # TODO
    #hyperparameters
    num_epochs = 10
    learning_rate = 0.1
    #initialize the model
    model = MyModel()
    #initialize the loss: categorical cross entropy
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    #initialize the optimizer: SGD with default parameters
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    #initialize lists for later visualization.
    train_losses = []
    test_losses = []
    test_accuracies = []

    #testing once before we begin
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    #check how model performs on train data once before we begin
    train_loss, _ = test(model, train_dataset, cross_entropy_loss)
    train_losses.append(train_loss)

    #train for num_epochs epochs
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

        #training (and checking in with training)
        epoch_loss_agg = []
        for input,target in train_dataset:
            train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)
        
        #track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))

        #testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    #visualize accuracy and loss for training and test data
    #note: accuracy of 35-40% is sufficient
    plt.figure()
    line1, = plt.plot(train_losses)
    line2, = plt.plot(test_losses)
    line3, = plt.plot(test_accuracies)
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1,line2, line3),("training","test", "test accuracy"))
    plt.show()

    
def train_step(model, input, target, loss_function, optimizer):
  #loss_object and optimizer_object are instances of respective tensorflow classes
  with tf.GradientTape() as tape:
    prediction = model(input)
    loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

def test(model, test_data, loss_function):
  #test over complete test data
  test_accuracy_aggregator = []
  test_loss_aggregator = []

  for (input, target) in test_data:
    prediction = model(input)
    sample_test_loss = loss_function(target, prediction)
    sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
    sample_test_accuracy = np.mean(sample_test_accuracy)
    test_loss_aggregator.append(sample_test_loss.numpy())
    test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

  test_loss = tf.reduce_mean(test_loss_aggregator)
  test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

  return test_loss, test_accuracy
  """

    """
    # train for n_epochs
    while epoch <= config['training']['epochs']:
        print("epoch: {}".format(epoch))
        epoch += 1
        batch = 0

        for real_image, real_label in train_ds:
            batch += 1

            with tf.GradientTape() as encoder_tape:
                # get initial latent approximations z0
                z0 = encoder(real_image, True)
                
                # optimize using the L-BFGS-B algorithm
                z_opt = lbfgs_opt(generator, res_net, real_image, real_label, z0)
                z_opt = z_opt.position

                # calculate encoder loss
                encoder_loss = euclidean_dist(z_opt, z0)
                encoder_losses.append(encoder_loss)

                # optimize encoder
                encoder_gradients = encoder_tape.gradient(encoder_loss, encoder.trainable_variables)
                optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))

            if batch % 50 == 0:
                print("batch {}".format(batch))
                print(encoder_loss)

                # Generate and save images
                image_z0 = generator(z0, real_label, False)
                image_z_opt = generator(z_opt, real_label, False)
                tf.keras.utils.save_img(config['data']['output_path'] + 'images/z_optimizer/epoch_' + str(epoch) + '_z0.png', image_z0[0])
                tf.keras.utils.save_img(config['data']['output_path'] + 'images/z_optimizer/epoch_' + str(epoch) + '_z_opt.png', image_z_opt[0])



        # visualize loss
        train_E_losses.append(np.mean(encoder_losses))

        plt.figure()
        line1 = plt.plot(train_E_losses)
        plt.xlabel("Training steps")
        plt.ylabel("Loss")
        plt.legend((line1),("E Loss"))
        plt.title("Optimized Encoder Loss")
        plt.savefig(config['data']['output_path'] + 'plots/opt_enc_loss.png')
        plt.close()


    # save model weights
    encoder.save_weights(config['data']['weight_path'] + 'opt_enc/opt_enc_weights')
    """
    

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
        train_Model(config)
    else:
        print("No valid model found")
