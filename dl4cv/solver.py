from random import shuffle
import numpy as np
import datetime
import torch
from torch.autograd import Variable
import os
import pickle
import time

class Solver(object):

    def __init__(self):
        self.history = {'train_loss_history': [],
                        'val_loss_history'  : []
                        }

        self.optim = []
        self.criterion = []
        self.training_time_s = 0
        self.stop_reason = ''

    def train(self, model, optim=None, loss_criterion=torch.nn.MSELoss(), num_epochs=10, max_train_time_s=None, start_epoch=0,
              train_loader=None, val_loader=None,
              log_after_iters=1, save_after_epochs=None,
              save_path='../saves/train', device='cpu'):

        model.to(device)

        if start_epoch == 0:
            self.optim = optim
            self.criterion = loss_criterion

        iter_per_epoch = len(train_loader)

        # Exponentially filtered training loss
        # Exponentially filtered losses
        train_loss_avg = 0
        best_val_loss = 0
        patience_counter = 0
        val_loss_avg = 0

        # Generate save folder
        save_path = os.path.join(save_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        os.makedirs(save_path)

        # Calculate the total number of minibatches for the training procedure
        n_iters = num_epochs*iter_per_epoch
        i_iter = 0

        t_start_training = time.time()

        print('Start training at epoch ' + str(start_epoch+1))

        # Do the training here
        for i_epoch in range(num_epochs):
            t_start_epoch = time.time()

            i_epoch += start_epoch

            # Set model to train mode
            model.train()

            for i_iter_in_epoch, batch in enumerate(train_loader):
                t_start_iter = time.time()

                i_iter += 1

                input, _ = batch

                # If the current minibatch does not have the full number of samples, skip it
                if len(input) < train_loader.batch_size:
                    print("Skipped batch")
                    continue

                input = input.to(device)

                # Forward pass
                output = model(input)

                loss = self.criterion(output, input)

                # Packpropagate and update weights
                model.zero_grad()
                loss.backward()
                optim.step()

                ## Save loss to history
                smooth_window_train = 100

                self.history['train_loss_history'].append(loss.item())
                train_loss_avg = (smooth_window_train-1)/smooth_window_train*train_loss_avg + 1/smooth_window_train*loss.item()

                if log_after_iters is not None and (i_iter % log_after_iters == 0):
                    print("Iteration " + str(i_iter) + "/" + str(n_iters) + "   Train loss: " + "{0:.3f}".format(loss.item()) + "   Avg: " + "{0:.3f}".format(train_loss_avg) + " - " + str(int((time.time()-t_start_iter)*1000)) + "ms")



            # Validate model
            print("Validate model after epoch " + str(i_epoch+1))

            # Set model to evaluation mode
            model.eval()

            num_val_batches = 0
            val_loss = 0

            for i, batch in enumerate(val_loader):
                num_val_batches += 1

                input, _ = batch

                input = input.to(device)

                output = model(input)

                loss = self.criterion(output, input)
                loss = loss.cpu().detach().numpy()
                # den = loss.shape[1]*loss.shape[2]*loss.shape[3]

                # loss = loss.sum(3).sum(2).sum(1)/den

                val_loss += loss.item()

            val_loss /= num_val_batches
            self.history['val_loss_history'].append(val_loss)

            smooth_window_val = 10

            if val_loss_avg == 0:
                val_loss_avg = val_loss
            else:
                val_loss_avg = (smooth_window_val - 1) / smooth_window_val * val_loss_avg + 1 / smooth_window_val * val_loss

            print("Epoch " + str(i_epoch+1) + '/' + "{0:.3f}".format(num_epochs) + ' Train loss: ' + "{0:.3f}".format(
                train_loss_avg) + '   Val loss: ' + "{0:.3f}".format(val_loss) + "   - Avg: " + "{0:.3f}".format(
                val_loss_avg) + "   - " + str(int((time.time() - t_start_epoch) * 1000)) + "ms")

            # self.history['lr_history'].append(self.lr)
            #
            # # Update learning rate
            # if i_epoch+1 % lr_decay_interval == 0:
            #     self.lr *= lr_decay
            #     print("Learning rate: " + str(self.lr))
            #     for i, _ in enumerate(optim.param_groups):
            #         optim.param_groups[i]['lr'] = self.lr

            # Early stopping
            # if patience is not None:
            #     if best_val_loss == 0 or val_loss_avg < best_val_loss:
            #         best_val_loss = val_loss_avg
            #         patience_counter = 0
            #     else:
            #         patience_counter += 1
            #
            #     if patience_counter > patience:
            #         print("Early stopping after " + str(i_epoch) + " epochs.")
            #         self.stop_reason = "Early stopping."
            #         break

            # Save model and solver
            if save_after_epochs is not None and ((i_epoch + 1) % save_after_epochs == 0):
                model.save(save_path + '/model' + str(i_epoch + 1))
                self.save(save_path + '/solver' + str(i_epoch + 1))
                model.to(device)

            # Stop if training time is over
            if max_train_time_s is not None and (time.time() - t_start_training > max_train_time_s):
                print("Training time is over.")
                self.stop_reason = "Training time over."
                break

        if self.stop_reason is "":
            self.stop_reason = "Reached number of specified epochs."

        self.training_time_s += time.time() - t_start_training

        # Save model and solver after training
        model.save(save_path + '/model' + str(i_epoch + 1))
        self.save(save_path + '/solver' + str(i_epoch + 1))

        print('FINISH.')


    def save(self, path):
        print('Saving solver... %s' % path)
        pickle.dump(self, open(path, 'wb'))
