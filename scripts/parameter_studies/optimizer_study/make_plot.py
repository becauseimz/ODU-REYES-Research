import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

# accuracy (for each filter_size configuration) of image patters for each 2D focal plane plot
xtick_labels = ['$x_{fp}$ vs $x^{\prime}_{fp}$', '$x_{fp}$ vs $y_{fp}$', '$x_{fp}$ vs $y^{\prime}_{fp}$', '$x^{\prime}_{fp}$ vs $y_{fp}$', '$x^{\prime}_{fp}$ vs $y^{\prime}_{fp}$', '$y^{\prime}_{fp}$ vs $y_{fp}$']
xdata = [0, 1, 2, 3, 4, 5]

acc_adam    = [0.9, 1.0, 1.0, 0.6, 1.0, 0.7 ]
acc_RMSprop = [0.8, 0.9, 1.0, 0.6, 1.0, 0.6 ]
acc_SGD     = [0.5, 1.0, 0.8, 0.2, 0.1, 0.7 ]
acc_Nadam   = [0.8, 0.8, 1.0, 0.6, 1.0, 0.7 ]
acc_Adamax  = [0.8, 1.0, 1.0, 0.5, 0.6, 0.7 ]

#plot data
plt.plot(xdata, acc_RMSprop, linestyle='', marker='s',mfc='b', ms=8, color='b',  label=r'optimizer: RMSprop')
plt.plot(xdata, acc_SGD, linestyle='', marker='o',mfc='g', ms=8, color='g',  label=r'optimizer: SGD')
plt.plot(xdata, acc_Nadam, linestyle='', marker='^',mfc='m', ms=8, color='m',  label=r'optimizer: Nadam')
plt.plot(xdata, acc_Adamax, linestyle='', marker='*',mfc='k', ms=8, color='k',  label=r'optimizer: Adamax')
plt.plot(xdata, acc_adam, linestyle='', marker='D',mfc='r', ms=8, color='r',  label=r'optimizer: adam')

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
