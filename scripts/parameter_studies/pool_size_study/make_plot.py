import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

# accuracy (for each filter_size configuration) of image patters for each 2D focal plane plot
xtick_labels = ['$x_{fp}$ vs $x^{\prime}_{fp}$', '$x_{fp}$ vs $y_{fp}$', '$x_{fp}$ vs $y^{\prime}_{fp}$', '$x^{\prime}_{fp}$ vs $y_{fp}$', '$x^{\prime}_{fp}$ vs $y^{\prime}_{fp}$', '$y^{\prime}_{fp}$ vs $y_{fp}$']
xdata = [0, 1, 2, 3, 4, 5]


acc_poolsize1  = [0.9, 0.1, 0.8, 0.4, 0.9, 0.7 ]
acc_poolsize2  = [0.8, 0.7, 1.0, 0.6, 1.0, 0.7 ]
acc_poolsize4  = [0.9, 0.8, 1.0, 0.6, 1.0, 0.7 ]
acc_poolsize6  = [0.9, 1.0, 1.0, 0.6, 1.0, 0.7 ]
acc_poolsize8  = [0.8, 1.0, 1.0, 0.5, 1.0, 0.6 ]

#plot data
plt.plot(xdata, acc_poolsize1, linestyle='', marker='o', mfc='b', ms=8, color='b', label=r'pool\_size=1')
plt.plot(xdata, acc_poolsize2, linestyle='', marker='^', mfc='g', ms=8, color='g', label=r'pool\_size=2')
plt.plot(xdata, acc_poolsize4, linestyle='', marker='s',mfc='k', ms=8, color='k',  label=r'pool\_size=4')
plt.plot(xdata, acc_poolsize6, linestyle='', marker='D',mfc='r', ms=8, color='r',  label=r'pool\_size=6')
plt.plot(xdata, acc_poolsize8, linestyle='', marker='*',mfc='m', ms=8, color='m',  label=r'pool\_size=8')

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
