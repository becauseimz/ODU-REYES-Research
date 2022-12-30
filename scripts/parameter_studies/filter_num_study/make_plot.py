import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

# accuracy (for each num_filter configuration) of image patters for each 2D focal plane plot
xtick_labels = ['$x_{fp}$ vs $x^{\prime}_{fp}$', '$x_{fp}$ vs $y_{fp}$', '$x_{fp}$ vs $y^{\prime}_{fp}$', '$x^{\prime}_{fp}$ vs $y_{fp}$', '$x^{\prime}_{fp}$ vs $y^{\prime}_{fp}$', '$y^{\prime}_{fp}$ vs $y_{fp}$']
xdata = [0, 1, 2, 3, 4, 5]
acc_numfil8  = [0.8, 0.6, 1.0, 0.5, 1.0, 0.7]  # using num_filter = 8
acc_numfil4  = [0.8, 0.6, 0.9, 0.6, 1.0, 0.7]  # using num_filter = 4
acc_numfil12 = [0.8, 0.6, 1.0, 0.6, 1.0, 0.7 ] # using num_filter = 12
acc_numfil24 = [0.8, 0.6, 1.0, 0.6, 1.0, 0.7]  #using num_filter = 24

#plot data
plt.plot(xdata, acc_numfil4, linestyle='', marker='o', mfc='b', ms=8, color='b', label=r'num\_filter=4')
plt.plot(xdata, acc_numfil8, linestyle='', marker='^', mfc='g', ms=9, color='g', label=r'num\_filter=8')
plt.plot(xdata, acc_numfil12, linestyle='', marker='s',mfc='r', ms=6, color='r', label=r'num\_filter=12')
plt.plot(xdata, acc_numfil24, linestyle='', marker='>',mfc='k', ms=8, color='k', label=r'num\_filter=24')

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
