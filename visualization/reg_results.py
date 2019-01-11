import numpy as np
import os
import pickle
import sys
import matplotlib.pyplot as plt

job_id = sys.argv[-1]
path = os.path.join(os.sep, 'home', 'user', 'project', 'run', job_id, 'logits_0')
with open(path, 'rb') as f:
    logits, labels = pickle.load(f)
repeated = np.tile(labels, [1, logits.shape[1]])
rmse_simple = np.sqrt(((logits[:, :, 0] - repeated) ** 2).mean())
pred = logits.mean(1)

print(np.corrcoef(labels[:, 0], pred[:, 0]))
idxs = np.argsort(labels[:, 0])
plt.plot(labels[:, 0][idxs], label='True label')
plt.plot(pred[:, 0][idxs], label='Prediction')
plt.ylabel('Survival time (years)')
plt.xlabel('Images sorted by true label')
plt.legend()
plt.show()
plt.plot(labels[:, 0][idxs], pred[:, 0][idxs])
plt.xlabel('True label (years)')
plt.ylabel('Prediction (years)')
plt.show()

rmse_averaged = np.sqrt(((pred - labels) ** 2).mean())
print(rmse_simple, '->', rmse_averaged)
