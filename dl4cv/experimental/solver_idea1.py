import datetime
import os
import time

import torch


class Solver(object):

    def __init__(self):
        self.history = {'train_loss_history': [],
                        'val_loss_history'  : [],
                        'kl_divergence_history': [],
                        'cov_history': []
                        }

        self.optim = []
        self.criterion = []
        self.training_time_s = 0
        self.stop_reason = ''

    def train(self, model, optim=None, loss_criterion=torch.nn.MSELoss(),
              num_epochs=10, max_train_time_s=None,
              train_loader=None, val_loader=None,
              log_after_iters=1, save_after_epochs=None,
              save_path='../saves/train', device='cpu', cov_penalty=0, beta=1,
              rec2_weight=1, rec3_weight=1, pos2_weight=1e4, pos3_weight=1e4,
              vel2_weight=2e4):

        model.to(device)

        start_epoch = len(self.history['val_loss_history'])

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

        # Do the training here
        for i_epoch in range(num_epochs):
            t_start_epoch = time.time()

            i_epoch += start_epoch

            # Set model to train mode
            model.train()

            for i_iter_in_epoch, batch in enumerate(train_loader):
                t_start_iter = time.time()

                i_iter += 1

                imgs_normal, imgs_mirrored, _ = batch

                # If the current minibatch does not have the full number of samples, skip it
                if len(imgs_normal) < train_loader.batch_size:
                    print("Skipped batch")
                    continue

                imgs_normal = imgs_normal.to(device)
                imgs_mirrored = imgs_mirrored.to(device)

                # Forward pass
                y2_pred_n, y3_pred_n, z_t_n, z_t_plus_1_n = model(imgs_normal[:, :-1])
                y2_pred_m, y3_pred_m, z_t_m, z_t_plus_1_m = model(imgs_mirrored[:, :-1])

                rec_loss_n2 = self.criterion(y2_pred_n, imgs_normal[:, -2].unsqueeze(dim=1))
                rec_loss_m2 = self.criterion(y2_pred_m, imgs_mirrored[:, -2].unsqueeze(dim=1))
                rec_loss_2 = rec2_weight * (rec_loss_n2 + rec_loss_m2)

                rec_loss_n3 = self.criterion(y3_pred_n, imgs_normal[:, -1].unsqueeze(dim=1))
                rec_loss_m3 = self.criterion(y3_pred_m, imgs_mirrored[:, -1].unsqueeze(dim=1))
                rec_loss_3 = rec3_weight * (rec_loss_n3 + rec_loss_m3)

                px2_loss = self.criterion(z_t_n[:, 0], -z_t_m[:, 0])
                py2_loss = self.criterion(z_t_n[:, 1], z_t_m[:, 1])

                px3_loss = self.criterion(z_t_plus_1_n[:, 0], -z_t_plus_1_m[:, 0])
                py3_loss = self.criterion(z_t_plus_1_n[:, 1], z_t_plus_1_m[:, 1])

                vx2_loss = self.criterion(z_t_n[:, 2], -z_t_m[:, 2])
                vy2_loss = self.criterion(z_t_n[:, 3], z_t_m[:, 3])

                p2_loss = pos2_weight * (px2_loss + py2_loss)
                p3_loss = pos3_weight * (px3_loss + py3_loss)
                v2_loss = vel2_weight * (vx2_loss + vy2_loss)

                loss = rec_loss_2 + \
                       rec_loss_3 + \
                       p2_loss + \
                       p3_loss + \
                       v2_loss

                # Back-propagate and update weights
                model.zero_grad()
                loss.backward()
                self.optim.step()

                # Save loss to history
                smooth_window_train = 10

                self.history['train_loss_history'].append(loss.item())
                train_loss_avg = (smooth_window_train-1)/smooth_window_train*train_loss_avg + 1/smooth_window_train*loss.item()

                if log_after_iters is not None and (i_iter % log_after_iters == 0):
                    print("Iteration " + str(i_iter) + "/" + str(n_iters) +
                          "   Train loss: " + "{0:.6f}".format(loss.item()) +
                          "   Avg train loss: " + "{0:.6f}".format(train_loss_avg) +
                          "   Rec loss 2: " + "{0:.6f}".format(rec_loss_2.item()) +
                          "   Rec loss 3: " + "{0:.6f}".format(rec_loss_3.item()) +
                          "   Pos loss 2: " + "{0:.6f}".format(p2_loss.item()) +
                          "   Pos loss 3: " + "{0:.6f}".format(p3_loss.item()) +
                          "   Vel loss 2: " + "{0:.6f}".format(v2_loss.item()) +
                          " - Time/iter " + str(int((time.time()-t_start_iter)*1000)) + "ms")

            # Validate model
            print("\nValidate model after epoch " + str(i_epoch+1) + '/' + str(num_epochs))

            # Set model to evaluation mode
            model.eval()

            num_val_batches = 0
            val_loss = 0

            for i, batch in enumerate(val_loader):
                num_val_batches += 1

                imgs_normal, imgs_mirrored, _ = batch

                imgs_normal = imgs_normal.to(device)
                imgs_mirrored = imgs_mirrored.to(device)

                # Forward pass
                y2_pred_n, y3_pred_n, z_t_n, z_t_plus_1_n = model(imgs_normal[:, :-1])
                y2_pred_m, y3_pred_m, z_t_m, z_t_plus_1_m = model(imgs_mirrored[:, :-1])

                rec_loss_n2 = self.criterion(y2_pred_n, imgs_normal[:, -2].unsqueeze(dim=1))
                rec_loss_m2 = self.criterion(y2_pred_m, imgs_mirrored[:, -2].unsqueeze(dim=1))
                rec_loss_2 = rec2_weight * (rec_loss_n2 + rec_loss_m2)

                rec_loss_n3 = self.criterion(y3_pred_n, imgs_normal[:, -1].unsqueeze(dim=1))
                rec_loss_m3 = self.criterion(y3_pred_m, imgs_mirrored[:, -1].unsqueeze(dim=1))
                rec_loss_3 = rec3_weight * (rec_loss_n3 + rec_loss_m3)

                px2_loss = self.criterion(z_t_n[:, 0], -z_t_m[:, 0])
                py2_loss = self.criterion(z_t_n[:, 1], z_t_m[:, 1])
                p2_loss = pos2_weight * (px2_loss + py2_loss)

                px3_loss = self.criterion(z_t_plus_1_n[:, 0], -z_t_plus_1_m[:, 0])
                py3_loss = self.criterion(z_t_plus_1_n[:, 1], z_t_plus_1_m[:, 1])
                p3_loss = pos3_weight * (px3_loss + py3_loss)

                vx2_loss = self.criterion(z_t_n[:, 2], -z_t_m[:, 2])
                vy2_loss = self.criterion(z_t_n[:, 3], z_t_m[:, 3])
                v2_loss = vel2_weight * (vx2_loss + vy2_loss)

                val_loss += rec_loss_2 + \
                            rec_loss_3 + \
                            p2_loss + \
                            p3_loss + \
                            v2_loss

            val_loss /= num_val_batches
            self.history['val_loss_history'].append(val_loss)

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
