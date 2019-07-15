import datetime
import os
import time

import torch

from dl4cv.utils import kl_divergence, time_left
from dl4cv.eval.eval_functions import generate_img_figure_for_tensorboardx
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from tensorboardX import SummaryWriter

class Solver(object):

    def __init__(self):
        self.history = {}

        self.optim = []
        self.criterion = []
        self.training_time_s = 0
        self.stop_reason = ''
        self.epoch = 0

    def train(
            self,
            model,
            train_config,
            dataset_config,
            tensorboard_path,
            optim=None,
            num_epochs=10,
            max_train_time_s=None,
            train_loader=None,
            val_loader=None,
            log_after_iters=1,
            save_after_epochs=None,
            save_path='../saves/train',
            device='cpu',
            target_var=1.,
            C_offset=0,
            C_max=20,
            C_stop_iter=1e5,
            gamma=100,
            log_reconstructed_images=True
    ):

        self.train_config = train_config
        self.dataset_config = dataset_config
        model.to(device)

        if self.epoch == 0:
            self.optim = optim

        iter_per_epoch = len(train_loader)
        print("Iterations per epoch: {}".format(iter_per_epoch))

        # Exponentially filtered training loss
        train_loss_avg = 0

        # Path to save model and solver
        save_path = os.path.join(save_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

        tensorboard_writer = SummaryWriter(os.path.join(tensorboard_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
                                           flush_secs=30)

        # Calculate the total number of minibatches for the training procedure
        n_iters = num_epochs*iter_per_epoch
        i_iter = 0

        print('Start training at epoch ' + str(self.epoch))
        t_start_training = time.time()

        self.C_offset = C_offset
        self.C_stop_iter = C_stop_iter
        self.gamma = gamma
        self.C_max = torch.autograd.Variable(torch.FloatTensor([C_max]))
        self.C_max = self.C_max.to(device)

        # Do the training here
        for i_epoch in range(num_epochs):
            self.epoch += 1
            print("Starting epoch {}".format(self.epoch))
            t_start_epoch = time.time()

            # Set model to train mode
            model.train()

            for i_iter_in_epoch, batch in enumerate(train_loader):
                t_start_iter = time.time()
                i_iter += 1

                x, y, question, _ = batch

                x = x.to(device)
                y = y.to(device)
                if question is not None:
                    question = question.to(device)

                # Forward pass
                y_pred, (mu, logvar) = model(x, question)

                # Compute losses
                reconstruction_loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction='sum').div(y.shape[0])
                total_kl_divergence, dim_wise_kld, mean_kld = kl_divergence(mu, logvar, target_var)

                C = torch.clamp(self.C_offset + self.C_max / self.C_stop_iter * i_iter, 0, self.C_max.data[0])

                loss = reconstruction_loss + gamma * (total_kl_divergence-C).abs()

                # Backpropagate and update weights
                model.zero_grad()
                loss.backward()
                self.optim.step()

                smooth_window_train = 10
                train_loss_avg = (smooth_window_train-1)/smooth_window_train*train_loss_avg + 1/smooth_window_train*loss.item()

                if log_after_iters is not None and (i_iter % log_after_iters == 0):
                    print("Iteration " + str(i_iter) + "/" + str(n_iters) +
                          "   C: {0:.2f}".format(C.item()) +
                          "   Reconstruction loss: " + "{0:.6f}".format(reconstruction_loss.item()),
                          "   KL loss: " + "{0:.6f}".format(total_kl_divergence.item()) +
                          "   Train loss: " + "{0:.6f}".format(loss.item()) +
                          "   Avg train loss: " + "{0:.6f}".format(train_loss_avg) +
                          " - Time/iter: " + str(int((time.time()-t_start_iter)*1000)) + "ms")

                    # plot_grad_flow(model.named_parameters())

                mus = mu.mean(dim=0).tolist()
                vars = logvar.exp().mean(dim=0).tolist()

                self.append_history({'train_loss': loss.item(),
                                     'total_kl_divergence': total_kl_divergence.item(),
                                     'kl_divergence_dim_wise': dim_wise_kld.tolist(),
                                     'reconstruction_loss': reconstruction_loss.item(),
                                     'posterior_mu': mus,
                                     'posterior_var': vars
                                     })

                # Add losses to tensorboard
                tensorboard_writer.add_scalar('Reconstruction_loss', reconstruction_loss.item(), i_iter)

                z_keys = ['C', 'Total_KL_loss']
                z_keys.extend(['z{}'.format(i) for i in range(dim_wise_kld.numel())])
                kls = [C.item(), total_kl_divergence.item()]
                kls.extend(dim_wise_kld.tolist())
                tensorboard_writer.add_scalars('KL_loss', dict(zip(z_keys, kls)), i_iter)

                z_keys = ['z{}'.format(i) for i in range(dim_wise_kld.numel())]
                tensorboard_writer.add_scalars('Posterior_means', dict(zip(z_keys, mus)), i_iter)
                tensorboard_writer.add_scalars('Posterior_variances', dict(zip(z_keys, vars)), i_iter)

                if log_reconstructed_images and os.getcwd()[:20] != '/home/felix.meissen':
                    f = generate_img_figure_for_tensorboardx(y, y_pred, question)
                    plt.show()  # don't log images on server
                    tensorboard_writer.add_figure('Reconstructed sample', f, i_iter)

            # Validate model
            print("\nValidate model after epoch " + str(self.epoch) + '/' + str(num_epochs))

            # Set model to evaluation mode
            model.eval()

            num_val_batches = 0
            val_loss = 0

            for i, batch in enumerate(val_loader):
                num_val_batches += 1

                x, y, question, _ = batch

                x = x.to(device)
                y = y.to(device)
                if question is not None:
                    question = question.to(device)

                y_pred, latent_stuff = model(x, question)

                current_val_loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction='sum').div(y.shape[0])

                val_loss += current_val_loss.item()

            val_loss /= num_val_batches

            self.append_history({'val_loss': val_loss})

            print('Avg Train Loss: ' + "{0:.6f}".format(train_loss_avg) +
                  '   Val loss: ' + "{0:.6f}".format(val_loss) +
                  "   - " + str(int((time.time() - t_start_epoch) * 1000)) + "ms" +
                  "   time left: {}\n".format(time_left(t_start_training, n_iters, i_iter)))

            # Save model and solver
            if save_after_epochs is not None and (self.epoch % save_after_epochs == 0):
                os.makedirs(save_path, exist_ok=True)
                model.save(save_path + '/model' + str(self.epoch))
                self.training_time_s += time.time() - t_start_training
                self.save(save_path + '/solver' + str(self.epoch))
                model.to(device)

            # Stop if training time is over
            if max_train_time_s is not None and (time.time() - t_start_training > max_train_time_s):
                print("Training time is over.")
                self.stop_reason = "Training time over."
                break

        if self.stop_reason is "":
            self.stop_reason = "Reached number of specified epochs."

        # Save model and solver after training
        os.makedirs(save_path, exist_ok=True)
        model.save(save_path + '/model' + str(self.epoch))
        self.training_time_s += time.time() - t_start_training
        self.save(save_path + '/solver' + str(self.epoch))

        print('FINISH.')

    def save(self, path):
        print('Saving solver... %s\n' % path)
        torch.save({
            'history': self.history,
            'epoch': self.epoch,
            'stop_reason': self.stop_reason,
            'training_time_s': self.training_time_s,
            'criterion': self.criterion,
            'optim_state_dict': self.optim.state_dict(),
            'train_config': self.train_config,
            'dataset_config': self.dataset_config
        }, path)

    def load(self, path, device, only_history=False):

        checkpoint = torch.load(path, map_location=device)

        if not only_history:
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.criterion = checkpoint['criterion']

        self.history = checkpoint['history']
        self.epoch = checkpoint['epoch']
        self.stop_reason = checkpoint['stop_reason']
        self.training_time_s = checkpoint['training_time_s']
        if 'train_config' in checkpoint.keys():
            self.train_config = checkpoint['train_config']
        if 'dataset_config' in checkpoint.keys():
            self.dataset_config = checkpoint['dataset_config']

    def append_history(self, hist_dict):
        for key in hist_dict:
            if key not in self.history:
                self.history[key] = [hist_dict[key]]
            else:
                self.history[key].append(hist_dict[key])

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
