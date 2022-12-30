import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

# accuracy (for each epoch configuration) of image patters for each 2D focal plane plot
xtick_labels = ['$x_{fp}$ vs $x^{\prime}_{fp}$', '$x_{fp}$ vs $y_{fp}$', '$x_{fp}$ vs $y^{\prime}_{fp}$', '$x^{\prime}_{fp}$ vs $y_{fp}$', '$x^{\prime}_{fp}$ vs $y^{\prime}_{fp}$', '$y^{\prime}_{fp}$ vs $y_{fp}$']
xdata = [0, 1, 2, 3, 4, 5]
acc_epoch10  = [0,   0.2, 0.2, 0.2, 0.1, 0.2]
acc_epoch50  = [0.4, 0.7, 1.0, 0.6, 0.9, 0.7]
acc_epoch100 = [0.8, 0.6, 1.0, 0.5, 1.0, 0.7]

#plot data
plt.plot(xdata, acc_epoch10, linestyle='', marker='o', mfc='b', ms=8, color='b', label='10 epochs')
plt.plot(xdata, acc_epoch50, linestyle='', marker='^', mfc='g', ms=8, color='g', label='50 epochs')
plt.plot(xdata, acc_epoch100, linestyle='', marker='s',mfc='r', ms=6, color='r', label='100 epochs')

# set tick labels, and rotate labels
plt.xticks(xdata, xtick_labels, fontsize=16)
plt.xticks(rotation=45)

#set axis limits
plt.ylim(-0.1, 1.1)

plt.ylabel('accuracy', fontsize=16)
plt.yticks(fontsize=14)
plt.title('Convolutional Neural Network Model:\n Accuracy of SHMS Optics Test Images', fontsize=20)
#auto-adjust layout and plot legend
plt.tight_layout()
plt.legend()
plt.grid(True)

plt.plot()

plt.show()
