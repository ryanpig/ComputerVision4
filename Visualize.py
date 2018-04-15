import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# evenly sampled time at 200ms intervals
#t = np.arange(0., 5., 0.2)
#a = np.array([1, 4, 2, 1])
#b = np.arange(1, 5)

def plot_two_arr(arr1,arr2):
    # debug
    print(len(arr1))
    print(len(arr2))
    #print(arrs[2])
    lenTr = len(arr1)
    lenTe = len(arr2)
    x = range(0, lenTr)

    plt.title('Accuracy of Training & Evaluation')
    plt.xlabel('The number of steps (/50)')
    plt.ylabel('Accuracy')
    #plt.xlim(xmin=2)
    a0, = plt.plot(x, arr1, 'r')
    a1, = plt.plot(x, arr2, 'g')
   # a2, = plt.plot(x, arrs[2], 'b')
    plt.legend((a0, a1), ('Acc_Train','Acc_Eval'))
    plt.show()
# Draw accuracy of training, evaluating dataset, and the loss of training in single plot.
def plot_acc_loss(arr1,arr2,loss):
    lenTr = len(arr1)
    lenTe = len(arr2)
    x = range(0, lenTr)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    a0, = ax1.plot(x,arr1, 'ro')
    a1, = ax1.plot(x,arr2, 'gx')

    a2, = ax2.plot(x,loss, 'b-')
    ax1.set_xlabel('The number of steps (/50)')
    ax1.set_ylabel('Accuracy', color='g')
    ax2.set_ylabel('Loss', color='b')

    plt.title('Accuracy of Training & Evaluation')
    plt.legend((a0, a1, a2), ('Acc_Train', 'Acc_Eval', 'Loss_Train'))
    plt.show()
    ax2 = ax1.twinx()

def plot_acc_bar(y,title_str):
    #x = range(len(y))
    #n, bins, patches = plt.hist(x,bins=10, normed=True, facecolor='green')
    x = np.arange(len(y))
    #plt.bar(x, height=[1, 2, 3])
    plt.bar(x, y)
    #plt.hist(x, normed=True, bins=4)
    #plt.xticks(x + .5, ['a', 'b', 'c']);
    # add a 'best fit' line
    #y = mlab.normpdf(bins, mu, sigma)
    #l = plt.plot(bins, y, 'r--', linewidth=1)
    plt.ylim(ymax=1)
    plt.xlabel('ModelNo.')
    plt.ylabel('Accuracy')
    plt.title(title_str)
    #plt.axis([40, 160, 0, 0.03])
    #plt.grid(True)

    plt.show()



a = [0.5,0.3,0.6]
b = [0.6,0.8,0.9]
c = [1.2,1.5,1.6]
# plot_two_arr(a, b)
#plot_acc_loss(a,b,c,)
#plot_acc_bar(np.arange(10))
