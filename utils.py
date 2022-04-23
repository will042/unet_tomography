
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

def input_data(stage='train', n_patches=100, patch_size=256, aug_mode = 'random_patches', plot = False, random_seed=1):
    '''
    Converts the images into TF tensors.
    stage: can be 'train' or 'test'; it reads the images from the corresponding folder
    n_patches: determines the number of pataches in random_patches aug_mode
    patch_size: dimensions of the square patches in pixels
    aug_mode: two augmentation modes: 'random_patches' and 'non-overlapping'
    plot: if True, plots the original segmentation map and spatial distribution of selected patches.
    random_seed: random seed in 'random_patches' augmentation mode
    
    output: 
    X: a TF tensor containing the patches of image with dimensions: [n_patches (or count),patch_size, patch_size, 1]
    Y: a TF tensor containing the patches of segmentation map with dimensions: [n_patches (or count),patch_size, patch_size, 3 (number of classes)]
    '''

    if stage=='train':
        random.seed(random_seed)
        imgcode=next(os.walk('data/train/image'))[2][0]
        image=np.asarray(Image.open(os.path.join('data/train/image', imgcode)))
        mask=np.asarray(Image.open(os.path.join('data/train/mask', imgcode)))
    elif stage=='test':
        random.seed(random_seed+1)
        imgcode=next(os.walk('data/test/image'))[2][0]
        image=np.asarray(Image.open(os.path.join('data/test/image', imgcode)))
        mask=np.asarray(Image.open(os.path.join('data/test/mask', imgcode)))
    else:
        raise ValueError("'stage' is not defined!")

    count = 0

    IMG_HEIGHT = 2043
    IMG_WIDTH = 2005
    IMG_CHANNELS = 1

    n_classes = 3

    if plot:
        fig,ax=plt.subplots(1)
        ax.imshow(mask*100,cmap='gray', vmin=0, vmax=255)


    if aug_mode=='random_patches':
        X=np.zeros((n_patches,patch_size,patch_size,IMG_CHANNELS),dtype=np.uint8)
        Y=np.zeros((n_patches,patch_size,patch_size,n_classes),dtype=np.uint8)

        while count<n_patches:

            upperleft_x=random.choice(range(IMG_HEIGHT-patch_size))
            upperleft_y=random.choice(range(IMG_WIDTH-patch_size))
            img=image[upperleft_x:upperleft_x+patch_size,upperleft_y:upperleft_y+patch_size]
            img2=mask[upperleft_x:upperleft_x+patch_size,upperleft_y:upperleft_y+patch_size]
            
            
            img=np.expand_dims(img,axis=-1)
            img2=np.expand_dims(img2,axis=-1)

            if np.max(img2)>0:
                X[count]=img
                for i in range(n_classes):
                    ind = np.where(img2==i)
                    Y[count,ind[0],ind[1],i]=1


                count+=1
                if plot:
                    rect=patches.Rectangle((upperleft_y,upperleft_x),patch_size,patch_size,linewidth=1,
                                    edgecolor='w',facecolor="none")
                    ax.add_patch(rect)


    elif aug_mode=='non-overlapping':
        max_patches=int(IMG_HEIGHT/patch_size)*int(IMG_WIDTH/patch_size)
        X=np.zeros((max_patches,patch_size,patch_size,IMG_CHANNELS),dtype=np.uint8)
        Y=np.zeros((max_patches,patch_size,patch_size,1),dtype=np.bool)
        for i in range(0,IMG_HEIGHT-patch_size,patch_size):
            for j in range(0,IMG_WIDTH-patch_size,patch_size):
                upperleft_x=j
                upperleft_y=i
                img=image[upperleft_x:upperleft_x+patch_size,upperleft_y:upperleft_y+patch_size]
                img2=mask[upperleft_x:upperleft_x+patch_size,upperleft_y:upperleft_y+patch_size]

                img=np.expand_dims(img,axis=-1)
                img2=np.expand_dims(img2,axis=-1)

                if np.max(img2)>0:
                    X[count]=img
                    Y[count]=img2
                    count+=1
                    if plot:
                        rect=patches.Rectangle((upperleft_y,upperleft_x),patch_size,patch_size,linewidth=1,
                                        edgecolor='w',facecolor="none")
                        ax.add_patch(rect)

        X=X[:count]
        Y=Y[:count]


    if plot:
        plt.title("Selected patches for "+stage.capitalize())
        plt.show() 


    return X,Y




def mIoU(mask, prediction, smooth=1):
    '''
    Calculates the Intersection of Union between two images
    mask and prediction dimentions: m x r x c x n 
    where m is the number of images, r is the number of rows, c is the number of columns, and n is the number of classes.

    Ref: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    '''
    mask = tf.cast(mask ,'float32')
    prediction = tf.cast(prediction ,'float32')

    intersection = K.sum(K.abs(mask * prediction), axis=[1,2,3])
    union = K.sum(mask,[1,2,3])+K.sum(prediction,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou




if __name__ == "__main__":

    x,y=input_data(stage='test',plot=True,n_patches=100,patch_size=256,aug_mode='random_patches')

    x = tf.convert_to_tensor(X)
    y = tf.convert_to_tensor(Y)

    with tf.compat.v1.Session() as sess:  print(mIoU(y,y).eval()) 


