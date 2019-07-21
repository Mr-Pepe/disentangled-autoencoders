import datetime
import os
import time

import torch

from dl4cv.utils import kl_divergence, time_left
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
import torch.nn.functional as F

from tensorboardX import SummaryWriter


class SolverDecoder(object):

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
            save_after_epochs=None,
            save_path='../saves/train',
            device='cpu',
            z_scale_factor=1.,
            overfit=False,
    ):

        self.train_config = train_config
        self.dataset_config = dataset_config
        model.to(device)

        if self.epoch == 0:
            self.optim = optim

        iter_per_epoch = len(train_loader)
        print("Iterations per epoch: {}".format(iter_per_epoch))

        # Path to save model and solver
        save_path = os.path.join(save_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

        tensorboard_writer = SummaryWriter(os.path.join(tensorboard_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
                                           flush_secs=30)

        # Calculate the total number of minibatches for the training procedure
        n_iters = num_epochs*iter_per_epoch
        i_iter = 0

        latent_names = dataset_config['latent_names']
        z_dim_decoder = train_config['z_dim_decoder']
        z_dim = z_dim_decoder - 1 if train_config['question'] else z_dim_decoder

        print('Start training at epoch ' + str(self.epoch))
        t_start_training = time.time()

        # Do the training here
        for i_epoch in range(num_epochs):
            self.epoch += 1
            print("Starting epoch {}".format(self.epoch))
            t_start_epoch = time.time()
            avg_train_loss = 0.

            # Set model to train mode
            model.train()

            for i_iter_in_epoch, batch in enumerate(train_loader):
                t_start_iter = time.time()
                i_iter += 1

                _, y, q, z = batch

                # z is the first frame of ground truth
                z = z[:, 0].float()
                latent_dim = z.shape[1]

                # Rescale values of z to be in a "KL-conform" range
                z *= z_scale_factor

                # Fill zeros in z with random normals
                z[:, len(latent_names):] = torch.randn((z.shape[0], z.shape[1] - len(latent_names)))

                if z_dim > latent_dim:
                    # Concatenate random normals to z
                    diff = z_dim - latent_dim
                    z = torch.cat((z, torch.randn((z.shape[0], diff))), dim=1)
                elif z_dim < latent_dim:
                    # Use only first few z as latent input
                    z = z[:, :z_dim]

                z = z.to(device)
                y = y.to(device)
                q = q.to(device)
                if q[0] > 0:
                    z = torch.cat((z, q.view(-1, 1)), dim=1)

                # Forward pass
                y_pred = model(z)

                # Compute loss
                # loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction='sum').div(y.shape[0])
                loss = F.smooth_l1_loss(y_pred, y)

                # Backpropagate and update weights
                model.zero_grad()
                loss.backward()
                self.optim.step()

                avg_train_loss += loss.item()

                tensorboard_writer.add_scalar('train_loss', loss.item(), i_iter)

                self.append_history({'train_loss': loss.item()})

            # Validate model
            print("\nValidate model after epoch " + str(self.epoch) + '/' + str(num_epochs))

            avg_train_loss /= iter_per_epoch

            # Set model to evaluation mode
            model.eval()

            num_val_batches = 0
            val_loss = 0

            for i, batch in enumerate(val_loader):
                num_val_batches += 1

                _, y, q, z = batch

                # z is the first frame of ground truth
                z = z[:, 0].float()
                latent_dim = z.shape[1]

                # Rescale values of z to be in a "KL-conform" range
                z *= z_scale_factor

                # Fill zeros in z with random normals
                z[:, len(latent_names):] = torch.randn((z.shape[0], z.shape[1] - len(latent_names)))

                if z_dim > latent_dim:
                    # Concatenate random normals to z
                    diff = z_dim - latent_dim
                    z = torch.cat((z, torch.randn((z.shape[0], diff))), dim=1)
                elif z_dim < latent_dim:
                    # Use only first few z as latent input
                    z = z[:, :z_dim]

                z = z.to(device)
                y = y.to(device)
                q = q.to(device)
                if q[0] > 0:
                    z = torch.cat((z, q.view(-1, 1)), dim=1)

                # Forward pass
                y_pred = model(z)

                # Compute loss
                # loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction='sum').div(y.shape[0])
                loss = F.smooth_l1_loss(y_pred, y)

                val_loss += loss.item()

                if save_after_epochs is not None and (self.epoch % save_after_epochs == 0) and overfit:
                    to_pil = transforms.ToPILImage()
                    f, axes = plt.subplots(1, 3)

                    y = y.detach().cpu()
                    y_pred = y_pred.detach().cpu()

                    # Plot ground truth
                    axes[0].imshow(to_pil(y[0]), cmap='gray')

                    # Plot prediction
                    axes[1].imshow(to_pil(y_pred[0]), cmap='gray')

                    # Plot Deviation
                    diff = abs(y_pred[0] - y[0])
                    axes[2].imshow(to_pil(diff), cmap='gray')

                    plt.show()

            val_loss /= num_val_batches

            self.append_history({
                'val_loss': val_loss,
                'avg_train_loss': avg_train_loss
            })

            tensorboard_writer.add_scalar('Avg train loss', avg_train_loss, i_iter)
            tensorboard_writer.add_scalar('val loss', val_loss, i_iter)

            print('Avg Train Loss: ' + "{0:.6f}".format(avg_train_loss) +
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
