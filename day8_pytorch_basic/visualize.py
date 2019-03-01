# bokeh
import numpy as np
from bokeh.plotting import figure, output_notebook

try:
    output_notebook()
    print('Available output Bokeh figure in notebook')
except Exception as e:
    print(e)

def mscatter(p, x, y, size=5, fill_color="orange" , alpha=0.5, marker="circle"):
    p.scatter(x, y, marker=marker, size=size,
        fill_color= fill_color, alpha=alpha) 

def scatter(x, y, title='', size=5, color='orange', height=600, width=600):
    p = figure(title=title)
    p.width = width
    p.height = height
    
    mscatter(p, x, y, size, color)
    return p

# matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt

def show_tensor_image(img, cmap='gray'):
    if len(img.shape) >= 3:
        img = torch.squeeze(img)
    img = img.numpy()
    show_image(img, cmap)

def show_image(img_2d, cmap='gray'):
    plt.imshow(img_2d, cmap=cmap)
    plt.show()