# This file contains utility functions for image processing
from PIL import Image, ImageOps
import numpy as np
from skimage.transform import resize
import os
import torchvision.transforms as transforms
import cv2
import matplotlib.image as mpimg
import random

categ_map = {'glioma':1, 'meningioma':2, 'notumor':3, 'pituitary':4}

def image_to_array(image_path, dimension):
    """Returns np.array representation of the image with the given dimension

    Args:
        image_path (string): path to the image
        dimension (int): final dimension of image array

    Returns:
        np.arary : np.array representation of image
    """
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    image_arr = resize(np.array(image), (dimension, dimension)) # resize to smaller dimension
    image.close()
    return image_arr.reshape(-1, dimension*dimension)

    # return resize(image_arr, (dimension,dimension)).reshape((1, dimension*dimension*3)).T

def generate_dataset():
    def helper(categ, train):
        subdir = "Training" if train else "Testing"
        subdir = "archive/" + subdir + "/" + categ + "/"
        files = os.listdir(subdir)
        arr = np.empty((0,64*64+1), dtype=np.float16)
        for file in files:
            if file.endswith(".jpg"):
                img_arr = image_to_array(subdir+file, dimension=64)
                row = np.insert(img_arr, 0, categ_map[categ])
                arr = np.row_stack((arr, row))
                # arr = np.append(arr, row, axis = 0)
        return arr
    """Generates train and test set

        Returns:
            list: list of train and test set as np.array
        """
    categ_path_train = os.listdir("archive/Training")
    train_set = np.empty((0,64*64+1), dtype=np.float16)
    for categ in categ_path_train:
        print("Processing", categ, "set for training")
        subarr = helper(categ, train = True)
        train_set = np.append(train_set, subarr, axis=0)
    
    categ_path_test = os.listdir("archive/Testing")
    test_set = np.empty((0,64*64+1), dtype=np.float16)
    for categ in categ_path_test:
        print("Processing", categ, "set for testing")
        subarr = helper(categ, train = False)
        test_set = np.append(test_set, subarr, axis=0)

    return [train_set, test_set]

def augment_brightness_camera_images(image):
    if len(image.shape)==3:
        image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    else:
        image1 = cv2.cvtColor(image,cv2.IMREAD_GRAYSCALE)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    if len(img.shape) == 3:
        rows,cols, ch = img.shape
    else:
        rows,cols = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    
    # Brightness 
    

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    # img = augment_brightness_camera_images(img)
    
    return img

def generate_transformed_images():
    if (len(os.listdir("archive/transformed/glioma")) != 0):
        print("Transformed images were already generated")
        return
    categories = os.listdir("archive/Testing")
    for categ in categories:
        files = os.listdir("archive/Testing/{}".format(categ))
        files = list(filter(lambda x: x.endswith(".jpg"), files))
        randomind = random.sample(range(0, len(files)), 10)
        random_files = list(map(lambda x: files[x], randomind))
        for file in random_files:
            image = mpimg.imread("archive/Testing/{}/{}".format(categ,file))
            img = transform_image(image,20,10,5)
            mpimg.imsave("archive/transformed/{}/{}".format(categ,file), img)

def generate_transformed_dataset():
    def helper(categ):
        subdir = "archive/transformed/{}/".format(categ)
        files = os.listdir(subdir)
        arr = np.empty((0,64*64+1), dtype=np.float16)
        for file in files:
            if file.endswith(".jpg"):
                img_arr = image_to_array(subdir+file, dimension=64)
                row = np.insert(img_arr, 0, categ_map[categ])
                arr = np.row_stack((arr, row))
                # arr = np.append(arr, row, axis = 0)
        return arr
    categ_path_trans = os.listdir("archive/transformed")
    trans_set = np.empty((0,64*64+1), dtype=np.float16)
    for categ in categ_path_trans:
        print("Processing", categ, "set for transformed set")
        subarr = helper(categ)
        trans_set = np.append(trans_set, subarr, axis=0)
    return trans_set

if __name__ == "__main__":
    # arr = image_to_array("archive/Testing/glioma/Te-gl_0010.jpg", dimension=64)
    # im = image_to_PIL("archive/Testing/glioma/Te-gl_0010.jpg")
    # im.save("demo2.jpg")
    # try:
    #     [train, test]  = [np.load("train.npy", allow_pickle=True), np.load("test.npy", allow_pickle=True)]
    # except FileNotFoundError:
    #     [train, test] = generate_dataset()
    #     np.save("train", train)
    #     np.save("test", test)
    # print("Training set: ", train.shape)
    # print("Testing set: ", test.shape)

    generate_transformed_images()

    trans_set = generate_transformed_dataset()
    np.save("trans", trans_set)
    print("Trans set: ", trans_set.shape)

    


# 1. function to convert to PIL Image format resized 
# 2. put all images in one array with an identifier 
#
