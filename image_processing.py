# This file contains utility functions for image processing
from PIL import Image, ImageOps
import numpy as np
from skimage.transform import resize
import os
import torchvision.transforms as transforms

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


if __name__ == "__main__":
    # arr = image_to_array("archive/Testing/glioma/Te-gl_0010.jpg", dimension=64)
    # im = image_to_PIL("archive/Testing/glioma/Te-gl_0010.jpg")
    # im.save("demo2.jpg")
    try:
        [train, test]  = [np.load("train.npy", allow_pickle=True), np.load("test.npy", allow_pickle=True)]
    except FileNotFoundError:
        [train, test] = generate_dataset()
        np.save("train", train)
        np.save("test", test)
    print("Training set: ", train.shape)
    print("Testing set: ", test.shape)
    


# 1. function to convert to PIL Image format resized 
# 2. put all images in one array with an identifier 
#
