# Multimodal-Interpretable-Framework

# prepare dataset
# Download and prepare datasets:
# Cityscapes: https://www.cityscapes-dataset.com/login/
# iSAID: https://captain-whu.github.io/iSAID/index.html

# generate clean segmentation results as attack targets
# use mmsegmentation to train a clean segmentation model
python tools/train.py configs/segmentation_config.py

# make instance-level adversarial attack
CUDA_VISIBLE_DEVICES=0 \
python tools/test_attack.py \
    configs/train_config.py \
    checkpoints/model.pth \
    --show-dir results/ \
    --data-root /path/to/dataset \
    --attack fgsm_m_mask_w2 \
    --dataset city \
    --n_iter 10 \
    --eps 128/255

# attack localization
CUDA_VISIBLE_DEVICES=0 accelerate launch train_att.py \
  --config configs/attDiffusion_352x352_fft.yaml \
  --num_epoch 100 \
  --batch_size 16 \
  --gradient_accumulate_every 1 \
  --results_folder results \
  --set \
  train_dataset.params.image_root=/path/to/train/images \
  train_dataset.params.gt_root=/path/to/train/masks \
  test_dataset.params.image_root=/path/to/val/images \
  test_dataset.params.gt_root=/path/to/val/masks

# run original class classifier
python Original_Class_Classifier.py

# run intent predictor
python Intent_Predictor.py

# run intent explanation
python Intent_Explanation.py

# evaluation
# evaluate the complete framework with DeepEval
# repo: https://github.com/confident-ai/deepeval
deepeval evaluate --config eval_config.yaml
