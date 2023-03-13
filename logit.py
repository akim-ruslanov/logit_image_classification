# This file will deal with fitting and testing logit model
from image_processing import generate_dataset
import numpy as np
from sklearn.linear_model import LogisticRegression

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
    categ = np.unique(train[:,0])
    if (categ.size != np.unique(test[:,0]).size):
        print("Test and Train set have unmatching number of categories")
    print("Training set: ", train.shape)
    print("Testing set: ", test.shape)

    logisticRegr = LogisticRegression(max_iter=1000000)

    logisticRegr.fit(train[:, 1:], train[:,0])

    pred = logisticRegr.predict(test[:,1:])

    pred_prob = logisticRegr.predict_proba(test[:,1:])

    print("Misclassification rate", sum(pred!=test[:,0])/test.shape[0])



        









