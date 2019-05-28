#help function


import h5py
import numpy as np
from PIL import Image

def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)
    

def load_hdf5(infile): #just load hdf5 format file.
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
        return f["image"][()]
    
def visualize(data,filename): # plot image
    assert (len(data.shape)==3) #height*width*channels
    img = None
    print('data shape : ',data.shape)
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    print(img)
    img.save(filename + '.png')
    print('file name : ',filename)
    return img


#group a set of images row per columns
def group_images(data,per_row):
    print('[group images func] prev data shape  :', data.shape)
    assert (data.shape[0]%per_row==0)
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow, make tensor, tensorflow backend
    print('[group images func] after data shape : ', data.shape)

    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    print('[group images func] first total image : ', totimg.shape)

    for i in range(0,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
        
    print('[group images func] final total image : ', totimg.shape)

    return totimg

def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs


#prepare the mask in the right shape for the Unet, Tensor
def masks_Unet(masks):
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,2))
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if  masks[i,j] == 0:
                new_masks[i,j,0]=1
                new_masks[i,j,1]=0
            else:
                new_masks[i,j,0]=0
                new_masks[i,j,1]=1
    return new_masks

def pred_to_imgs(pred, patch_h, patch_w, mode = 'original'):
    assert(len(pred.shape) ==3)
    assert(pred.shape[2] ==2) #binary?
    #shape[0] = number of patch
    #shape[1] = each pixel
    pred_imgs = np.empty((pred.shape[0], pred.shape[1]))
    
    if mode == 'original':
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_imgs[i,pix] = pred[i,pix,1]
    elif mode == 'threshold':
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1] >= 0.5:
                    pred_imgs[i,pix] =1
                else:
                    pred_imgs[i,pix] =0
    else:
        print ("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit(-1)
    pred_imgs = np.reshape(pred_imgs,(pred_imgs.shape[0],1, patch_h, patch_w))
    return pred_imgs
