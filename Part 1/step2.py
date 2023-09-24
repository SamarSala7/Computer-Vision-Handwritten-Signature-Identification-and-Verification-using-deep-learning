import cv2
import numpy as np
from glob import glob
import argparse
from helpers import *
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import time


class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.test_image_path = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []


    def trainModel(self):
        """
        This method contains the entire module
        required for training the bag of visual words model

        Use of helper functions will be extensive.

        """

        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path)
        # extract SIFT Features from each image
        label_count = 0
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            print ("Computing Features for ", word)
            for im in imlist:
                # cv2.imshow("im", im)
                # cv2.waitKey()
                self.train_labels = np.append(self.train_labels, label_count)
                kp, des = self.im_helper.features(im)
                self.descriptor_list.append(des)

            label_count += 1


        # perform clustering
        self.bov_helper.formatND(self.descriptor_list)
        self.bov_helper.cluster()
        self.bov_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list)

        # show vocabulary trained
        self.bov_helper.plotHist()

        self.bov_helper.standardize()
        self.bov_helper.train(self.train_labels)

        #print("Train Accuracy = " + str((label_count/self.trainImageCount) * 100))


    def recognize(self,test_img, test_image_path=None):

        """
        This method recognizes a single image
        It can be utilized individually as well.


        """

        kp, des = self.im_helper.features(test_img)
        # print kp
        print(des.shape)

        # generate vocab for test image
        vocab = np.array( [[ 0 for i in range(self.no_clusters)]])
        vocab = np.array(vocab, 'float32')
        test_ret = self.bov_helper.kmeans_obj.predict(des)
        # print test_ret

        # print vocab
        for each in test_ret:
            vocab[0][each] += 1

        #print (vocab)

        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)
        # predict the class of the image
        lb = self.bov_helper.clf.predict(vocab)
        # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb



    def testModel(self):
        correctClassifications = 0
        self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)

        predictions = []

        for word, imlist in self.testImages.items():
            print("processing ", word)
            for im in imlist:
                print(im.shape)
                cl = self.recognize(im)
                print(cl)
                predictions.append({
                    'image':im,
                    'class':cl,
                    'object_name':self.name_dict[str(int(cl[0]))]
                    })

                if(self.name_dict[str(int(cl[0]))]==word):
                    correctClassifications = correctClassifications + 1

        #print("Test Accuracy = " + str((correctClassifications/self.testImageCount) * 100))
        # print('-'*50)
        # print(predictions[20:30])
        # print('-'*50)

    def test_image(self):
        
        correctClassifications = 0
        self.testImages1, self.testImageCount1 = self.file_helper.getFiles(self.test_image_path)

        predictions = []

        for word, imlist in self.testImages1.items():
            #print("processing ", word)
            for im in imlist:
                #print(im.shape)
                cl = self.recognize(im)
                print(cl)
                predictions.append({
                    'image':im,
                    'class':cl,
                    'object_name':self.name_dict[str(int(cl[0]))]
                    })

                if(self.name_dict[str(int(cl[0]))]==word):
                    correctClassifications = correctClassifications + 1

        #print('-'*50)
        #print("Signature belongs to = " + str(predictions[]['object_name']))
        #print('-'*50)
        #print("Test Accuracy = " + str((correctClassifications/self.testImageCount1) * 100))
        #print(predictions)

        return predictions

    #def print_vars(self):
    #   pass

    def prep_img(self):
        self.testImages1, self.testImageCount1 = self.file_helper.getFiles(self.test_image_path)

        grays = []

        for word, imlist in self.testImages1.items():
            for im in imlist:
                # print('-'*50)
                # print(im.shape)
                # print('-'*50)
                #gray = im[:, :, 0]
                img = cv2.resize(im, dsize=(527, 390), interpolation=cv2.INTER_CUBIC)
                # x = img.reshape(img,-1)
                #img = np.expand_dims(img,axis=0)
                #print(img.shape)
                grays.append(img)
        return grays



if __name__ == '__main__':

    # parse cmd args
    parser = argparse.ArgumentParser(
            description=" Bag of visual words example"
        )
    parser.add_argument('--train_path', default="Part 1\\Train", action="store", dest="train_path")
    parser.add_argument('--test_path', default="Part 1\\Test", action="store", dest="test_path")
    #
    parser.add_argument('--test_image_path', default="Part 1\\Test_image", action="store", dest="test_image_path")
    #
    args = vars(parser.parse_args())
    print(args)

    bov = BOV(no_clusters=100)

    #set training paths
    bov.train_path = args['train_path']
    # set testing paths
    bov.test_path = args['test_path']
    #train the model
    # start_train = time.time()
    bov.trainModel()
    # end_train = time.time()
    # test model
    # start_test = time.time()
    # bov.testModel()
    # end_test = time.time()

    bov.test_image_path = args['test_image_path']
    predic = bov.test_image()
    images = bov.prep_img()

    print('-'*50)
    arr = np.array(images)
    #arr = np.expand_dims(arr,axis=-1)
    print(arr.shape)
    print('-'*50)
    # print(predic)
    #print(len(predic))

    # print(f'Train time = {(end_train-start_train):.2f} seconds, Testing time = {(end_test-start_test):.2f} seconds.')
    #print('-'*50)
    ypreds=[]

    for j in range(len(predic)):
        ypreds.append(predic[j]['object_name'])
        print("Signature belongs to = " + str(ypreds[j]))
    print('-'*50)
    # print(len(ypreds))
    # print('-'*50) 

    # for i in range(len(predic)):
    #     if predic[i]['object_name'] == 'PersonA':
    #        loaded_model = load_model('my_model0')
    #        y_pred = loaded_model.predict(images)
    #     elif predic[i]['object_name'] == 'PersonB':
    #        loaded_model = load_model('my_model1')
    #        y_pred = loaded_model.predict(images)

    #     elif predic[i]['object_name'] == 'PersonC':
    #        loaded_model = load_model('my_model2')
    #        y_pred = loaded_model.predict(images)

    #     elif predic[i]['object_name'] == 'PersonD':
    #        loaded_model = load_model('my_model3')
    #        y_pred = loaded_model.predict(images)

    #     else:
    #        loaded_model = load_model('my_model4')
    #        y_pred = loaded_model.predict(images)   
 
    # loaded_model = load_model('my_model0')
    # y_pred = loaded_model.predict(images)
    

    for x , y in zip(arr,ypreds):
        #x.reshape
        if(ypreds=='PersonA'):
          loaded_model = load_model('my_model0')
          y_pred = loaded_model.predict(x)

        elif(ypreds=='PersonB'):
          loaded_model = load_model('my_model1')
          y_pred = loaded_model.predict(x)

        elif(ypreds=='PersonC'):
          loaded_model = load_model('my_model2')
          y_pred = loaded_model.predict(x)

        elif(ypreds=='PersonD'):
          loaded_model = load_model('my_model3')
          y_pred = loaded_model.predict(x)

        else:
              loaded_model = load_model('my_model4')
              y_pred = loaded_model.predict(x)







    print('-'*50)
    print(y_pred)
    print('-'*50)
    print("End")

    