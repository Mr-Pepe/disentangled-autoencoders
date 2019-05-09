import pickle
import numpy as np
import matplotlib.pyplot as plt



def show_solver_history(path):
    solver = pickle.load(open(path, 'rb'))

    print("Stop reason: %s" %solver.stop_reason)
    print("Stop time: %fs" %solver.training_time_s)

    train_loss = np.array(solver.history['train_loss_history'])
    val_loss = np.array(solver.history['val_loss_history'])

    f, (ax1, ax2, ax3) = plt.subplots(3,1)

    ax1.plot(train_loss)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Train loss")

    ax2.plot(val_loss)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation loss")

    ax3.plot(train_loss)
    ax3.plot(val_loss)
    # ax3.plot(np.arange(0,len(val_loss))*460,np.convolve(val_loss, np.ones((15,)) / 15, mode='valid'))

    plt.show()
