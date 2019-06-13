import matplotlib.pyplot as plt
import numpy as np
from dl4cv.solver import Solver

path = '../../saves/train20190611073202/solver100'


solver = Solver()
solver.load(path, only_history=True)

print("Stop reason: %s" % solver.stop_reason)
print("Stop time: %fs" % solver.training_time_s)

train_loss = np.array(solver.history['train_loss'])
val_loss = np.array(solver.history['val_loss'])
kl_divergence = np.array(solver.history['kl_divergence'])
reconstruction_loss = np.array(solver.history['reconstruction_loss'])

f, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1.plot(train_loss)
ax1.plot(np.linspace(1, len(train_loss), len(val_loss)), val_loss)
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Train/Val loss")

ax2.plot(kl_divergence)
ax2.set_xlabel("Iterations")
ax2.set_ylabel("KL Divergence")

ax3.plot(reconstruction_loss)
ax3.set_xlabel("Iterations")
ax3.set_ylabel("Reconstruction Loss")

plt.show()
