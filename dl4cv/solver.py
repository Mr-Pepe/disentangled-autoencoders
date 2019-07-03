import datetime
import os
import time

import torch

from dl4cv.utils import kl_divergence, time_left


class Solver(object):

    def __init__(self):
        self.history = {'train_loss': [],
                        'val_loss': [],
                        'total_kl_divergence': [],
                        'kl_divergence_dim_wise': [],
                        'reconstruction_loss': [],
                        'beta': []
                        }

        self.optim = []
        self.criterion = []
        self.training_time_s = 0
        self.stop_reason = ''
        self.beta = 0
        self.epoch = 0

    def train(
            self,
            model,
            config,
            tensorboard_writer,
            optim=None,
            loss_criterion=torch.nn.MSELoss(),
            num_epochs=10,
            max_train_time_s=None,
            train_loader=None,
            val_loader=None,
            log_after_iters=1,
            save_after_epochs=None,
            save_path='../saves/train',
            device='cpu',
            cov_penalty=0,
            beta=1,
            beta_decay=1,
            patience=128,
            loss_weighting=False,
            loss_weight_ball=2.
    ):

        self.config = config
        model.to(device)

        if self.epoch == 0:
            self.optim = optim
            self.criterion = loss_criterion
            self.beta = beta

        iter_per_epoch = len(train_loader)
        print("Iterations per epoch: {}".format(iter_per_epoch))

        # Exponentially filtered training loss
        train_loss_avg = 0

        # Path to save model and solver
        save_path = os.path.join(save_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

        # Calculate the total number of minibatches for the training procedure
        n_iters = num_epochs*iter_per_epoch
        i_iter = 0

        best_recon_loss = 1e10
        n_bad_iters = 0

        t_start_training = time.time()

        print('Start training at epoch ' + str(self.epoch))
        t_start = time.time()

        # Do the training here
        for i_epoch in range(num_epochs):
            self.epoch += 1
            print("Starting epoch {}, Beta: {}".format(self.epoch, self.beta))
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

                # If using Loss weighting, create a weighting mask
                if loss_weighting:
                    loss_weight_mask = torch.where(y > 1e-3, y * loss_weight_ball, torch.ones_like(y))

                # Forward pass
                y_pred, latent_stuff = model(x, question)

                # Compute losses
                cov = torch.zeros(1, device=device)
                total_kl_divergence = torch.zeros(1, device=device)
                reconstruction_loss = self.criterion(y_pred, y)

                # When using loss weight, multiply the rec_loss with the weight mask and reduce afterwards
                if loss_weighting:
                    reconstruction_loss = reconstruction_loss * loss_weight_mask
                    reconstruction_loss = reconstruction_loss.mean() / loss_weight_mask.mean()

                # KL-loss if latent_stuff contains mu and logvar
                if len(latent_stuff) == 2:
                    mu, logvar = latent_stuff
                    total_kl_divergence, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                loss = reconstruction_loss + cov_penalty * cov + self.beta * total_kl_divergence

                # Backpropagate and update weights
                model.zero_grad()
                loss.backward()
                self.optim.step()

                # Reduce beta
                if reconstruction_loss.item() < best_recon_loss:
                    best_recon_loss = reconstruction_loss.item()
                    n_bad_iters = 0
                else:
                    n_bad_iters += 1

                if n_bad_iters >= patience:
                    self.beta *= beta_decay
                    n_bad_iters = 0

                smooth_window_train = 10
                train_loss_avg = (smooth_window_train-1)/smooth_window_train*train_loss_avg + 1/smooth_window_train*loss.item()

                if log_after_iters is not None and (i_iter % log_after_iters == 0):
                    print("Iteration " + str(i_iter) + "/" + str(n_iters) +
                          "   Reconstruction loss: " + "{0:.6f}".format(reconstruction_loss.item()),
                          "   KL loss: " + "{0:.6f}".format(total_kl_divergence.item()) +
                          "   Train loss: " + "{0:.6f}".format(loss.item()) +
                          "   Avg train loss: " + "{0:.6f}".format(train_loss_avg) +
                          " - Time/iter: " + str(int((time.time()-t_start_iter)*1000)) + "ms")

                self.append_history({'train_loss': loss.item(),
                                     'total_kl_divergence': total_kl_divergence.item(),
                                     'kl_divergence_dim_wise': dim_wise_kld.tolist(),
                                     'reconstruction_loss': reconstruction_loss.item(),
                                     'beta': self.beta     # Save beta every iteration to multiply with kl div
                                     })

                # Add losses to tensorboard
                tensorboard_writer.add_scalar('kl_loss', total_kl_divergence.item(), i_iter)
                tensorboard_writer.add_scalar('reconstruction_loss', reconstruction_loss.item(), i_iter)
                tensorboard_writer.add_scalar('train_loss', loss.item(), i_iter)
                z_keys = ['z{}'.format(i) for i in range(dim_wise_kld.numel())]
                tensorboard_writer.add_scalars('kl_loss_dim_wise',  dict(zip(z_keys, dim_wise_kld.tolist())), i_iter)
                tensorboard_writer.add_scalar('beta', self.beta)

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

                # If using Loss weighting, create a weighting mask
                if loss_weighting:
                    loss_weight_mask = torch.where(y > 1e-3, y * loss_weight_ball, torch.ones_like(y))

                y_pred, latent_stuff = model(x, question)

                current_val_loss = self.criterion(y, y_pred)

                # When using loss weight, multiply the rec_loss with the weight mask and reduce afterwards
                if loss_weighting:
                    current_val_loss = current_val_loss * loss_weight_mask
                    current_val_loss = current_val_loss.mean() / loss_weight_mask.mean()

                val_loss += current_val_loss.item()

            val_loss /= num_val_batches

            self.append_history({'val_loss': val_loss})

            print('Avg Train Loss: ' + "{0:.6f}".format(train_loss_avg) +
                  '   Val loss: ' + "{0:.6f}".format(val_loss) +
                  "   - " + str(int((time.time() - t_start_epoch) * 1000)) + "ms\n" +
                  "   time left: {}".format(time_left(t_start, n_iters, i_iter)))

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
            'beta': self.beta,
            'optim_state_dict': self.optim.state_dict(),
            'config': self.config
        }, path)

    def load(self, path, device, only_history=False):

        checkpoint = torch.load(path, map_location=device)

        if not only_history:
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.criterion = checkpoint['criterion']

        self.history = checkpoint['history']
        self.epoch = checkpoint['epoch']
        self.beta = checkpoint['beta']
        self.stop_reason = checkpoint['stop_reason']
        self.training_time_s = checkpoint['training_time_s']
        if 'config' in checkpoint.keys():
            self.config = checkpoint['config']

    def append_history(self, hist_dict):
        for key in hist_dict:
            self.history[key].append(hist_dict[key])
