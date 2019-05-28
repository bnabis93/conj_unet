import os, shutil
import sys
import h5py
import numpy as np
from PIL import Image
sys.path.insert(0, './lib/')
from help_functions import write_hdf5

# explicit data path 

#DRIVE training data path
original_img_train_path = './DRIVE/DRIVE/training/images/'
ground_truth_img_train_path = './DRIVE/DRIVE/training/1st_manual/'
border_masks_imgs_train_path = './DRIVE/DRIVE/training/mask/'

#DRIVE test data path
original_img_test_path = './DRIVE/DRIVE/test/images/'
ground_truth_img_test_path = './DRIVE/DRIVE/test/1st_manual/'
border_masks_imgs_test_path = './DRIVE/DRIVE/test/mask/'

#DRIVE PATH
#original_path = '/content/gdrive/My Drive/data/DRIVE_datasets/DRIVE/DRIVE'
#base_path = './'
#dataset_dir_path = os.path.join(base_path,'DRIVE_datasets_training_testing')

dataset_dir_path = './DRIVE_datasets_training_testing/'
#train_dir = os.path.join(base_path, 'train')
#valid_dir = os.path.join(base_path, 'validation')
#test_dir = os.path.join(base_path, 'test')

'''
#make dir
if os.path.isdir(train_dir) == False:
    os.mkdir(train_dir)
else:
    print('already exist the folder in this path : {}'.format(train_dir))
    
if os.path.isdir(valid_dir) == False:
    os.mkdir(valid_dir)
else:
    print('already exist the folder in this path : {}'.format(valid_dir))

if os.path.isdir(test_dir) == False:
    os.mkdir(test_dir)
else:
    print('already exist the folder in this path : {}'.format(test_dir))
'''    
#first, you should masking the each folder's image.
'''


DIR TREE

base_path  - train datasets
          |
          |- validation datasets
          |
          |- test datasets
          
          
original_path  - train 
              |
              |- test
              |
              |- masked train
              |
              |- masked test
          
'''


# DRIVE image information
num_imgs = 20
channels = 3
img_height = 584
img_width = 565

  
def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
    
    imgs = np.empty((num_imgs,img_height,img_width,channels))
    groundTruth = np.empty((num_imgs,img_height,img_width))
    border_masks = np.empty((num_imgs,img_height,img_width))
    
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print ("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            
            #corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print ("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            
            #corresponding border masks
            border_masks_name = ""
            if train_test=="train":
                border_masks_name = files[i][0:2] + "_training_mask.gif"
            elif train_test=="test":
                border_masks_name = files[i][0:2] + "_test_mask.gif"
            else:
                print ("specify if train or test!!")
                exit()
            print ("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)
            print(b_mask,'\n')
            
    print(border_masks.shape)
          
    print ("imgs max: " +str(np.max(imgs)))
    print ("imgs min: " +str(np.min(imgs)))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print ("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (num_imgs,channels,img_height,img_width))
    groundTruth = np.reshape(groundTruth,(num_imgs,1,img_height,img_width))
    border_masks = np.reshape(border_masks,(num_imgs,1,img_height,img_width))
    assert(groundTruth.shape == (num_imgs,1,img_height,img_width))
    assert(border_masks.shape == (num_imgs,1,img_height,img_width))
    return imgs, groundTruth, border_masks
    

if not os.path.exists(dataset_dir_path):
    os.makedirs(dataset_dir_path)
#getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_img_train_path,ground_truth_img_train_path,border_masks_imgs_train_path,"train")
print ("saving train datasets")
write_hdf5(imgs_train, dataset_dir_path + "DRIVE_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_dir_path + "DRIVE_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train,dataset_dir_path + "DRIVE_dataset_borderMasks_train.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(original_img_test_path,ground_truth_img_test_path,border_masks_imgs_test_path,"test")
write_hdf5(imgs_test,dataset_dir_path + "DRIVE_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_dir_path + "DRIVE_dataset_groundTruth_test.hdf5")
write_hdf5(border_masks_test,dataset_dir_path + "DRIVE_dataset_borderMasks_test.hdf5")
