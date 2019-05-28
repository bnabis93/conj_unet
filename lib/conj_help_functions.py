
def show_on_jupyter(img,color= None,title=None):
    import matplotlib.pyplot as plt
    import cv2
    """Show img on jupyter notebook. No return Value
    
    You should check the img's color space.
    I just consider about RGB color space & 1 ch color space(like green ch, gray space, ...)
    
    using matplotlib
    
    Parameters
    ----------
    img : 2-D Array
        numpy 2-D array
        opencv / sklearn / plt are avaliable.
        float / uint8 data type.
        
    color : string
        'gray' or 'None'
        'gray' means that img has a 1 ch.
        'None' means that img has a RGB ch.
        (default: None)
        
    title : string
        decide img's title
        (default : None)
        
    Returns
    -------
        No return value.
    
    Example
    -------
    >>> img = cv2.imread(img_path)
    >>> show_on_jupyter(img)
    
    img has a 1 ch
    >>> img = cv2.imread(img_path)
    >>> show_on_jupyter(img,'gray')
    """
    if color == 'gray':
        plt.axis("off")
        plt.title(title)
        plt.axis('off')
        plt.figure(figsize=(10, 10))
        plt.imshow(img,cmap=color)
        plt.show()
    elif color == None:
        plt.axis("off")
        plt.title(title)
        plt.axis('off')
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        print("Gray or None")

def load_data(data_path):
    
    imgs = np.empty((NUM_IMG,1,HEIGHT_IMG,WIDTH_IMG))
    cnt = 0
    for path, subdirs, files in os.walk(data_path): #list all files, directories in the path
        for i in range(len(files)):
            tempImg = Image.open(data_path + files[i])
            #tempImg = np.asarray(tempImg)
            #print(tempImg.shape)
            w, h = tempImg.size
            print(w,h)
            print('w,h : ', w,'x',h)
            if (w == 700 and h ==380):
                print("Hi")
                tempImg = np.asarray(tempImg)
                tempImg = tempImg[np.newaxis,:,:]
                imgs[cnt] = tempImg
                cnt += 1
            
    return imgs

def group_plot(data,row,col):
    fig=plt.figure(figsize=(13, 13))
    columns = col
    rows = row
    for num in range(len(data)):
        img = data[num]
        fig.add_subplot(rows, columns, num+1)
        plt.imshow(img,cmap='gray')
    plt.show()

