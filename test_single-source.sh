# --------------------- REQUIRED: Modify for each dataset and/or experiment ---------------------

# Set test source file (test_tgt is optional, can be used to compute CER and WER of the predicted output)
test_src="sample_data/testing/test_src1.txt"

# Set experiment parameters
expt_folder="my_expt_singlesource/"

dynet_mem=1000 # Memory in MB available for testing

params="--pretrain_dec --pretrain_s2s --pretrain_enc --pointer_gen --coverage --diag_loss 2"
trained_model_name="my_trained_model"

# ------------------------------END: Required experimental settings------------------------------


# Load the trained model and get the predicted output on the test set (add --dynet-gpu for using GPU)
python postcorrection/multisource_wrapper.py \
--dynet-mem $dynet_mem \
--dynet-autobatch 1 \
--test_src1 $test_src \
$params \
--single \
--vocab_folder $expt_folder/vocab \
--output_folder $expt_folder \
--load_model $expt_folder"/models/"$trained_model_name \
--testing
