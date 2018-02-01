import matplotlib.pyplot as plt

def Show(img):
    #ax = plt.axes([0,0,1,1])
    plt.axis('off')
    plt.tight_layout()
    img = plt.imshow(img, interpolation="nearest")
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

def Title(msg):
    plt.title(msg, fontsize=16)
