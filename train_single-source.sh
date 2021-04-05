# --------------------- REQUIRED: Modify for each dataset and/or experiment ---------------------

# Set pretraining, training and development set files
pretrain_src="sample_dataset/postcorrection/pretraining/pretrain_src1.txt"

train_src="sample_dataset/postcorrection/training/train_src1.txt"
train_tgt="sample_dataset/postcorrection/training/train_tgt.txt"

dev_src="sample_dataset/postcorrection/training/dev_src1.txt"
dev_tgt="sample_dataset/postcorrection/training/dev_tgt.txt"

# Set experiment parameters
expt_folder="my_expt_singlesource/"

dynet_mem=3000 # Memory in MB available for training

params="--pretrain_dec --pretrain_s2s --pretrain_enc --pointer_gen --coverage --diag_loss 2"
pretrained_model_name="my_pretrained_model"
trained_model_name="my_trained_model"

# ------------------------------END: Required experimental settings------------------------------



# Create experiment directories
mkdir $expt_folder
mkdir $expt_folder/debug_outputs
mkdir $expt_folder/models
mkdir $expt_folder/outputs
mkdir $expt_folder/pretrain_logs
mkdir $expt_folder/pretrain_models
mkdir $expt_folder/train_logs
mkdir $expt_folder/vocab


# Denoise outputs for pretraining
python utils/denoise_outputs.py \
--train_src1 $train_src \
--train_tgt $train_tgt \
--input $pretrain_src \
--output $pretrain_src'.denoised'

pretrain_tgt=$pretrain_src'.denoised'


# Create character vocabulary for the post-correction model
python postcorrection/create_vocab.py \
--src1_files $train_src $dev_src \
--tgt_files $train_tgt $dev_tgt \
--output_folder $expt_folder/vocab


# Pretrain the model (add --dynet-gpu for using GPU)
# See postcorrection/opts.py for all the options
python postcorrection/multisource_wrapper.py \
--dynet-mem $dynet_mem \
--dynet-autobatch 1 \
--pretrain_src1 $pretrain_src \
--pretrain_tgt $pretrain_tgt \
$params \
--single \
--vocab_folder $expt_folder/vocab \
--output_folder $expt_folder \
--model_name $pretrained_model_name \
--pretrain_only


# Load the pretrained model and train the model using manually annotated training data (add --dynet-gpu for using GPU)
# See postcorrection/opts.py for all the options
python postcorrection/multisource_wrapper.py \
--dynet-mem $dynet_mem \
--dynet-autobatch 1 \
--train_src1 $train_src \
--train_tgt $train_tgt \
--dev_src1 $dev_src \
--dev_tgt $dev_tgt \
$params \
--single \
--vocab_folder $expt_folder/vocab \
--output_folder $expt_folder \
--load_model $expt_folder"/pretrain_models/"$pretrained_model_name \
--model_name $trained_model_name \
--train_only
