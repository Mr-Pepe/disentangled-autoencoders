import datetime
import torch
import os
import time

from dl4cv.utils import kl_divergence, time_left


class Solver(object):

    def __init__(self):
        self.history = {'train_loss': [],
                        'val_loss'  : [],
                        'kl_divergence': [],
                        'reconstruction_loss': []
                        }

        self.optim = []
        self.criterion = []
        self.training_time_s = 0
        self.stop_reason = ''

    def train(self, model, optim=None, loss_criterion=torch.nn.MSELoss(),
              num_epochs=10, max_train_time_s=None,
              train_loader=None, val_loader=None,
              log_after_iters=1, save_after_epochs=None,
              save_path='../saves/train', device='cpu', cov_penalty=0, beta=1):

        model.to(device)

        start_epoch = len(self.history['val_loss'])

        if start_epoch == 0:
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

        t_start_training = time.time()

        print('Start training at epoch ' + str(start_epoch+1))
        t_start = time.time()

        # Do the training here
        for i_epoch in range(num_epochs):
            print("Starting epoch {}".format(start_epoch + i_epoch + 1))
            t_start_epoch = time.time()

            i_epoch += start_epoch

            # Set model to train mode
            model.train()

            for i_iter_in_epoch, batch in enumerate(train_loader):
                t_start_iter = time.time()

                i_iter += 1

                x, y, question, _ = batch

                # If the current minibatch does not have the full number of samples, skip it
                if len(x) < train_loader.batch_size:
                    print("Skipped batch, len(x): {}, batch_size: {}".format(
                        len(x), train_loader.batch_size
                    ))
                    continue

                x = x.to(device)
                y = y.to(device)
                if question is not None:
                    question = question.to(device)

                # Forward pass
                y_pred, latent_stuff = model(x, question)

                cov = torch.zeros(1, device=device)
                total_kl_divergence = torch.zeros(1, device=device)
                reconstruction_loss = self.criterion(y_pred, y)

                # KL-loss if latent_stuff contains mu and logvar
                if len(latent_stuff) == 2:
                    mu, logvar = latent_stuff
                    total_kl_divergence, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                elif len(latent_stuff) == 1:
                    z = latent_stuff[0]

                    # Calculate covariance
                    z = z[:, :, 0, 0]
                    z_mean = torch.mean(z, dim=0)
                    cov = 1/len(batch) * torch.mm((z - z_mean).transpose(0, 1), (z-z_mean))
                    cov = cov - torch.diag(torch.diag(cov)).to(device)
                    cov = torch.sum(cov*cov)

                loss = reconstruction_loss + cov_penalty * cov + self.beta * total_kl_divergence

                # Packpropagate and update weights
                model.zero_grad()
                loss.backward()
                self.optim.step()

                # Save loss to history
                smooth_window_train = 10

                self.history['train_loss'].append(loss.item())
                train_loss_avg = (smooth_window_train-1)/smooth_window_train*train_loss_avg + 1/smooth_window_train*loss.item()

                self.history['kl_divergence'].append(total_kl_divergence.item())
                self.history['reconstruction_loss'].append(reconstruction_loss.item())

                if log_after_iters is not None and (i_iter % log_after_iters == 0):
                    print("Iteration " + str(i_iter) + "/" + str(n_iters) +
                          "   Reconstruction loss: " + "{0:.6f}".format(reconstruction_loss.item()),
                          "   KL loss: " + "{0:.6f}".format(total_kl_divergence.item()) +
                          "   Train loss: " + "{0:.6f}".format(loss.item()) +
                          "   Avg train loss: " + "{0:.6f}".format(train_loss_avg) +
                          " - Time/iter: " + str(int((time.time()-t_start_iter)*1000)) + "ms" +
                          "   time left: {}".format(time_left(t_start, n_iters, i_iter)))

            # Validate model
            print("\nValidate model after epoch " + str(i_epoch+1) + '/' + str(num_epochs))

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

                val_loss += self.criterion(y, y_pred).item()

            val_loss /= num_val_batches
            self.history['val_loss'].append(val_loss)

            print('Avg Train Loss: ' + "{0:.6f}".format(train_loss_avg) +
                  '   Val loss: ' + "{0:.6f}".format(val_loss) +
                  "   - " + str(int((time.time() - t_start_epoch) * 1000)) + "ms\n")

            # Save model and solver
            if save_after_epochs is not None and ((i_epoch + 1) % save_after_epochs == 0):
                os.makedirs(save_path, exist_ok=True)
                model.save(save_path + '/model' + str(i_epoch + 1))
                self.training_time_s += time.time() - t_start_training
                self.save(save_path + '/solver' + str(i_epoch + 1))
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
        model.save(save_path + '/model' + str(i_epoch + 1))
        self.training_time_s += time.time() - t_start_training
        self.save(save_path + '/solver' + str(i_epoch + 1))

        print('FINISH.')

    def save(self, path):
        print('Saving solver... %s\n' % path)
        torch.save({
            'history': self.history,
            'stop_reason': self.stop_reason,
            'training_time_s': self.training_time_s,
            'criterion': self.criterion,
            'beta': self.beta,
            'optim_state_dict': self.optim.state_dict()
        }, path)

    def load(self, path, only_history=False):

        checkpoint = torch.load(path)

        if not only_history:
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.criterion = checkpoint['criterion']

        self.history = checkpoint['history']
        self.stop_reason = checkpoint['stop_reason']
        self.training_time_s = checkpoint['training_time_s']
