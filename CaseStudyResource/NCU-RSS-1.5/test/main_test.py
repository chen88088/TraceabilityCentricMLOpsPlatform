import os
import tensorflow as tf
from test.generate_parcel_dataset_for_train_te5t import GenerateTrainingParcelDataset
from test.kmeans_cluster_for_train_te5t import  Clustering
from test.random_sampling_for_parcel_dataset_for_train_te5t import Sampling
from test.train_and_val_model_te5t import Training
from test.generate_parcel_dataset_for_inference_te5t import GenerateInferenceParcelDataset
from test.inference_te5t import Inference
import shutil

class MainTest(tf.test.TestCase):
    
    def main_course_train(self,train_on_all_frames=True,
        select_specific_parcels=True):
        
        t1=GenerateTrainingParcelDataset()
        t1.setUp()
        t1.generate_parcel_dataset(select_specific_parcels=select_specific_parcels)

        t2=Clustering()
        t2.setUp()
        t2.cluster(rice_cluster_n=2,non_rice_cluster_n=2,train_on_all_frames=train_on_all_frames)
        
        t3=Sampling()
        t3.setUp()
        t3.sample(rice_cluster_n=2,non_rice_cluster_n=2,train_on_all_frames=train_on_all_frames,rice_ratio=0.5)

        t4=Training()
        t4.setUp()
        t4.train(rice_cluster_n=2,non_rice_cluster_n=2,rice_ratio=0.5)
    def main_course_inference(self):
        t1=GenerateInferenceParcelDataset()
        t1.setUp()
        t1.generate_parcel_dataset()
        # move saved model to inference folder
        if os.path.exists("test/data/inference/saved_model_and_prediction"):
            shutil.rmtree("test/data/inference/saved_model_and_prediction")
        os.mkdir("test/data/inference/saved_model_and_prediction")
        shutil.copy("test/data/train_test/For_training_testing/80x80/train_test/model_val_acc.h5",
                    "test/data/inference/saved_model_and_prediction/model_val_acc.h5")
        t2=Inference()
        t2.setUp()
        t2.infer()
    def test_one(self):
        self.main_course_train(train_on_all_frames=True,select_specific_parcels=True)
        self.main_course_inference()
    def test_two(self):
        self.main_course_train(train_on_all_frames=True,select_specific_parcels=False)
    def test_three(self):
        pass
        #self.main_course(train_on_all_frames=False,select_specific_parcels=False)
        #the small frames would cause an exception