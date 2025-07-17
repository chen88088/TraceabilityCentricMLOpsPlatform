import glob
from src.preprocessing.generate_parcel_dataset_based_on_complete_frame_aerial_image import generate_parcel_dataset
import tensorflow as tf
import os
class GenerateTrainingParcelDataset(tf.test.TestCase):
    def setUp(self):
        self.Data_root_folder_path = "./test/data/train_test"
        self.shape = (80, 80)

    def generate_parcel_dataset(self,select_specific_parcels):
        generate_parcel_dataset(
            shape=self.shape, dataset_root_folder_path=self.Data_root_folder_path, select_specific_parcels=select_specific_parcels,
            inference=False,
            NRG_png_path="./test/data/train_test/NRG_png",
            parcel_mask_path="./test/data/train_test/parcel_mask",
            selected_parcel_mask_path="./test/data/train_test/selected_parcel_mask",
            band=3
            )

        # assert parcel_nirrga exists
        parcel_nirrga_dir = os.path.join(self.Data_root_folder_path, "For_training_testing",
                                         "%dx%d" % (self.shape[0], self.shape[1]), "parcel_NIRRGA")
        print(parcel_nirrga_dir)
        self.assertEqual(True, os.path.isdir(parcel_nirrga_dir))

        # assert number of parcels in parcel_nirrga is equal to rows in gt_label.txt
        frames_path=glob.glob(parcel_nirrga_dir+'/*')
        for frame_path in frames_path:
            parcels=glob.glob(frame_path+'/*.npy')
            gt_label_txt=glob.glob(frame_path+'/*label.txt')
            with open(gt_label_txt[0]) as fh:
                lines=fh.readlines()
                self.assertEqual(len(lines),len(parcels))
    


