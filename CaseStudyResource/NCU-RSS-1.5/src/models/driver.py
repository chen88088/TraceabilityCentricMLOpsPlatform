"""
the arguments may need to be changed

optimizer_learning_rate:
    the learning rate of optimizer

BATCH_SIZE:
    batch size

random_brightness_max_delta:
    the effect of random brightness on each data, 1.3 model use 0.3,
    setting it to 0 means do not do any random brightness in the data augmentation part.

test_only:
    currently, used to decide train&test the model or test the model only.
    if True => train the model, and then do the testing.
    if False => load the model, test the model only

save_prediction :
    if True => In testing stage, show the parcel-based kappa,
        generate and show parcel-based segmentation result and save the result image.
    if False => In testing stage, show the parcel-based kappa only,
        do not generate and show parcel-based segmentation result and save the result image,

"""
import logging
from src.models import parcel_based_CNN_models
from src.models import parcel_based_CNN


# TODO reduce complexity, too many arguments and local variables
def main(arguments: parcel_based_CNN.Arguments = None,
         data_shape=None,
         round_number=None,
         saving_model_dir_path_front_part=None,
         optimizer_learning_rate=0.00001,
         lr_d=False,
         training_ds_frame_dataset=None,
         validation_ds_frame_dataset=None,
         EPOCH_N=10,
         Data_root_folder_path="./data/train_test",
         select_specific_parcels=False,
         saved_model_folder="./data/inference/saved_model_and_prediction",
         inference_NRG_png_path="./data/inference/NRG_png",
         inference_parcel_mask_path="./data/inference/parcel_mask_path",
         training_parcel_mask_path='./data/train_test/parcel_mask'
         ):
    """
        the arguments may need to be changed

        EPOCH_N: int
            the number of epochs for training
    """

    # build model
    model, model_name = parcel_based_CNN_models.pretrained_VGG16BNConv_GAP(
        input_shape=data_shape,
        ConvBlocks_BN=False,
        DP_rate=0.3,
        conv_blocks_trainable=True,
        use_GMP=False,
        use_DenseV0=False,
        Dense1_DP_rate=0,
        Dense2_DP_rate=0)

    # show the model
    model.summary()

    # affect the name of model
    model_name = model_name + "_NIRRG"
    if arguments.per_image_standardization is True:
        model_name = model_name + "_PIS"
    if arguments.random_brightness_max_delta != 0:
        model_name = model_name + \
                     f"_RB{arguments.random_brightness_max_delta}"

    if arguments.preprocess_input is True:
        model_name = model_name + "_0725PI"

    # set the parameters for model training

    saving_model_dir_path = saving_model_dir_path_front_part + '/train_test'
    training = parcel_based_CNN.Training(
        saving_model_dir_path=saving_model_dir_path,
        data_shape=data_shape,
        Round_number=round_number,
        EPOCH_N=EPOCH_N,
        BATCH_SIZE=arguments.BATCH_SIZE,
        optimizer_learning_rate=optimizer_learning_rate,
        training_ds_frame_dataset=training_ds_frame_dataset,
        validation_ds_frame_dataset=validation_ds_frame_dataset
    )

    # set the parameters for the model compile, such as optimizer
    model, model_name = training.model_compile(
        model,
        model_name,
        test_only=arguments.test_only,
        optimizer_name="Adam",
        lr=optimizer_learning_rate,
        focal_loss=False)

    arguments.model = model
    if arguments.inference:
        parcel_based_CNN.inference(training_inform_obj=training, arguments=arguments,
                                   Data_root_folder_path=Data_root_folder_path, saved_model_folder=saved_model_folder,inference_NRG_png_path=inference_NRG_png_path,inference_parcel_mask_path=inference_parcel_mask_path)

    else:
        # create tensorflow_dataset object, including training dataset and
        # validation dataset
        training.create_tensorflow_dataset(arguments)
        if arguments.test_only is False:  # Training

            history_obj, total_min = training.train(lr_d=lr_d)  # Train the model

            # show the information in training process
            history_list = [history_obj.history]
            # print("\n\n history_obj.history:{}".format(history_obj.history))
            parcel_based_CNN.show_history_acc_loss(
                saving_png_dir_path=saving_model_dir_path + '/val_acc.png', history_list=history_list,
                target_monitor="val_acc", network_name=model_name, time_cost=total_min)
            
            #　TODO: JERRY CHANHE HERE
            # 返回模型和训练历史
            return model, history_obj

        else:  # do the Testing only
            parcel_based_CNN.test_saved_models(training_inform_obj=training, arguments=arguments, max_val_acc=.9,
                                               Data_root_folder_path=Data_root_folder_path,
                                               select_specific_parcels=select_specific_parcels,
                                               training_parcel_mask_path=training_parcel_mask_path)
            
            #　TODO: JERRY CHANHE HERE
            # 如果只做测试，返回模型（可以选择是否返回 history_obj）
            return model, None
