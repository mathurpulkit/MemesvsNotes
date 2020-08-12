import webbrowser
import matplotlib.pyplot as plt
import copy

def reportgen(filename="test", network=[784,16,16,10], epochs=100, learn_rate=0.05,
              train_acc=1, test_acc=1, cost_f=1, batch_size=32, lamda=0, note=""):
    f = open(filename+".html", "w")
    f.writelines("<HTML>\n<HEAD>\n<title>" + "Report: " + str(filename) + "</title>")
    f.writelines("\n<style>h1 {text-align: center;}</style>\n</HEAD>\n")
    f.writelines("\n<BODY>")
    f.writelines("<H1> Report Name: " + filename + "</H1>\n")
    f.writelines("<H3>Neural Network Layer Structure is: "+str(network)+"</H3>\n")
    f.writelines("<H3>Batch Size used in training is: " + str(batch_size) + "</H3>\n")
    f.writelines("<H3>Number of EPOCHS done are: " + str(epochs) + "</H3>\n")
    f.writelines("<H3>Learning Rate used is: " + str(learn_rate) + "</H3>\n")
    f.writelines("<H3>Train set accuracy is: " + str(train_acc*100) + "%</H3>\n")
    f.writelines("<H3>Test set accuracy is: " + str(test_acc*100) + "%</H3>\n")
    f.writelines("<H3>Lambda used is: " + str(lamda) + "</H3>\n")
    f.writelines("<H3>Final Cost is: " + str(cost_f) + "</H3>\n")
    if note != "":
        f.writelines("<H3>Special Notes: "+note+"</H3>\n")
    f.writelines("<H3>Cost Function per EPOCH Graph is: </H3>\n")
    f.writelines('<center><img src="'+filename+'.png" alt="Cost Function per EPOCH Graph"></center>\n')
    f.writelines("<H3>Loss Function per Mini Batch Graph is: </H3>\n")
    f.writelines('<center><img src="' + filename +"b"+ '.png" alt="Cost Function Graph"></center>\n')
    f.writelines("<H3> Rate of Change of Loss Function per Mini-Batch Graph is: </H3>\n")
    f.writelines('<center><img src="' + filename +"d"+ '.png" alt="Rate of change of Cost Function Graph"></center>\n')
    f.writelines("<H3> Accuracy of Train set after some EPOCHS is: </H3>\n")
    f.writelines('<center><img src="' + filename +"a"+ '.png" alt="Accuracy Graph"></center>\n')
    f.writelines("\n</BODY>\n</HTML>\n")
    f.close()
    webbrowser.open(filename+".html")
    return


def imagegen(imgname, costs_epoch, costs_batch, accuracies):
    plt.plot(costs_epoch)
    plt.ylabel("Cost")
    plt.xlabel("No of EPOCHS")
    plt.savefig(imgname + ".png")  # saves image
    plt.clf()
    plt.plot(costs_batch)
    plt.ylabel("Loss")
    plt.xlabel("No of Mini-Batches")
    plt.savefig(imgname + "b" + ".png")
    plt.clf()
    plt.plot(derivative(costs_batch))
    plt.ylabel("Rate of Change of Loss")
    plt.xlabel("No of Mini-Batches")
    plt.savefig(imgname + "d" + ".png")
    plt.clf()
    axes = plt.gca()
    axes.set_ylim([0.5,1])
    plt.plot(accuracies)
    plt.ylabel("Accuracy on Train set")
    plt.xlabel("No of EPOCHS")
    plt.savefig(imgname + "a" + ".png")
    return

def derivative(x):
    y = copy.deepcopy(x)
    a = []
    for z in range(1,len(y)):
        a.append(y[z] - y[z-1])
    #print(a)
    return a