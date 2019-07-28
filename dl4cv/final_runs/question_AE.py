from dl4cv.train import train
from dl4cv.eval.eval import evaluate
from dl4cv.utils import Config, str2bool

import argparse

SAVE_PATH = '../../saves/Question_AE'
DATA_PATH = '../../../datasets/ball'
EVAL_DATA_PATH = '../../../datasets/evalDataset'

config = Config({

    'use_cuda': True,

    # Training continuation
    'continue_training':   False,      # Specify whether to continue training with an existing model and solver
    'model_path': '../saves/Question_AE/model60',
    'solver_path': '../saves/Question_AE/solver60',

    # Data
    'data_path': DATA_PATH,   # Path to the parent directory of the image folder
    'eval_data_path': EVAL_DATA_PATH,
    'load_data_to_ram': False,
    'dt': 1,                            # Frame rate at which the dataset got generated
    'do_overfitting': False,            # Set overfit or regular training
    'num_train_regular':    8192,       # Number of training samples for regular training
    'num_val_regular':      512,        # Number of validation samples for regular training
    'num_train_overfit':    256,        # Number of training samples for overfitting test runs
    'len_inp_sequence': 5,              # Length of training sequence
    'len_out_sequence': 1,              # Number of generated images

    'num_workers': 4,                   # Number of workers for data loading

    # Hyper parameters
    'max_train_time_s': None,
    'num_epochs': 600,                  # Number of epochs to train
    'batch_size': 64,
    'learning_rate': 5e-4,
    'betas': (0.9, 0.999),              # Beta coefficients for ADAM
    'target_var': 1,                 # Target variance for the kl loss
    'C_offset': 100,
    'C_max': 100,
    'C_stop_iter': 5e4,
    'gamma': 0,
    'beta': 0,

    # Model parameters
    'z_dim_encoder': 6,
    'z_dim_decoder': 7,
    'use_physics': False,
    'use_question': True,

    # Logging
    'log_interval': 20,           # Number of mini-batches after which to print training loss
    'save_interval': 10,         # Number of epochs after which to save model and solver
    'save_path': SAVE_PATH,
    'log_reconstructed_images': False,  # Show a reconstructed sample after every epoch
    'tensorboard_log_dir': '../../tensorboard_log/',


    ######### EVAL ##########################

    'analyze_dataset': True,            # Plot positions of the desired datapoints
    'show_solver_history': True,        # Plot losses of the training
    'show_latent_variables': True,      # Show the latent variables for the desired datapoints
    'show_model_output': True,          # Show the model output for the desired datapoints
    'eval_correlation': True,           # Plot the correlation between the latent variables and ground truth
    'latent_variable_slideshow': True,   # Create a slideshow varying over all latent variables
    'print_training_config': True,       # Print the config that was used for training the model
    'latent_walk_gifs': True,
    'walk_over_question': True,
    'eval_disentanglement': True,       # Evaluate disentanglement according to the metric from the BetaVAE paper.
    'mutual_information_gap': True,     # Evaluate disentanglement according to the MIG score

    'num_samples': 2000,                # Use the whole dataset if none for latent variables
    'num_show_images': 10,              # Number of outputs to show when show_model_output is True

    'question': True,

    'epoch': None,                                  # Use last model and solver if epoch is none
})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=False, type=str2bool, help='Train model')
    parser.add_argument('--eval', default=False, type=str2bool, help='Evaluate model')

    args = parser.parse_args()

    if args.train:
        train(config)

    config.update({
        'use_cuda': False
    })

    if args.eval:
        evaluate(vars(config))
