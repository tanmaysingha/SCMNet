# SCMNet: Shared Context Mining Network for Real-time Semantic Segmentation
This is an official site for SCMNet model. Relying on the dual-branch encoder design, this model proposed a shared-branch encoder network in which deep and shallow branches share their knowledge while propagating contextual details to the next layers. We also introduce a new module, called "Context mining Module" (CMM) which refines the contextual deatils after every shared point. Our proposed Deep Shallow Feature Fusion Module (DS-FFM) provides better object localization and context assimilation while generating segmented output. Model performance is evaluated on three publicly available benchmarks- Cityscapes, BDD100K and CamVid datasets. SCMNet can handle high resolution input images with less memory footprint. To compare our model performance with other existing semantice segmentation models, we also trained Deeplab, FAST-SCNN, ContextNet models under the same system configuration. Due to the large size of the network, we replace standard convolution layers by depth-wise separable convolution layers in DeepLab. We can either use Xception (X-65) or MobileNetV2 DCNN as backbone in DeepLab model. In our paper, we presented the results of all these models. Our proposed model SCMNet produces better results than many existing realtime semantic segmentation models which has less than 5 million parameters. We achieve 66.5% validation accuracy on Cityscapes dataset and 51.2% validation accuracy on BDD100K dataset whilst having only 1.2 Million parameters. The proposed model produces the state-of-the-art result on Camvid validation and test sets while having less parameters. It produces 78.6% and 71.3% validation and test accuracy respectively. In this repository, we uploaded the design of the porposed model and the existing models which are trained under the same system configuration. 

## Datasets
For this research work, we have used Cityscapes, BDD100K and CamVid datasets.
* Cityscapes - To access this benchmark, user needs an account. https://www.cityscapes-dataset.com/downloads/ 
* BDD100K - To access this benchmark, user needs an account. https://doc.bdd100k.com/download.html     
* CamVid - To access this benchmark, visit this link: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

## Metrics
To understand the metrics used for model performance evaluation, please  refer here: https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results

## Transfer Learning
For performance comparison, we trained few off-line and real-time existing models under same configuration and compared their performance with SCMNet. Some existing models require the use of ImageNet pretrained DCNNs to initialize their weights. For instance, in DeepLab, we use Xception as backbone network. We used pre-trained Xception model for DeepLab model.

## Requirements for Project
* TensorFlow 2.1
  * This requires CUDA >= 10.1
  * TensorRT will require an NVIDIA GPU with Tensor Cores.
  * Horovod framework (for effective utilization of resources and speed up GPUs)
* Keras 2.3.1
* Python >= 3.7

## Results
We trained our model with different input resolutions for different dataset. Cityscapes provides 1024 * 2048 px resolution images. We mainly focus full resolution of cityscapes images. For CamVid dataset, we use 640 * 896 px resolution altough original image size is 720 * 960 px. Similarly, we use 768 * 1280 px resolution input images for BDD100K dataset although original size of input image is 720 * 1280 px. For Cityscapes and BDD100K datasets, we use 19 classes, however for Camvid dataset we trained the model with 11 classes (suggested by the literature). 

### Complete pipeline of SCMNet
![pipeline](https://github.com/tanmaysingha/SCMNet/blob/main/figures/SCMNet_pipeline.png?raw=true)
  
### SCMNet prediction on Cityscapes test images
![Cityscapes_test_set](https://github.com/tanmaysingha/SCMNet/blob/main/figures/Cityscapes_Test_predictions.png?raw=true)  

### All models prediction on Citycapes validation set
![Cityscapes_all_models](https://github.com/tanmaysingha/SCMNet/blob/main/figures/Cityscapes_val_predictions.png?raw=true)

### All models prediction on BDD100K validation set
![BDD100K_all_models](https://github.com/tanmaysingha/SCMNet/blob/main/figures/BDD100K_val_predictions.png?raw=true)

### All models prediction on CamVid validation set
![CamVid_all_models](https://github.com/tanmaysingha/SCMNet/blob/main/figures/Camvid_val_predictions.png?raw=true)

### SCMNet Performance
<b><u>IoU Over Classes on Validation Set of Cityscapes</b></u>

classes       |  IoU  |   nIoU
--------------|-------|---------
road          | 0.967 |    nan
sidewalk      | 0.759 |    nan
building      | 0.902 |    nan
wall          | 0.531 |    nan
fence         | 0.503 |    nan
pole          | 0.475 |    nan
traffic light | 0.542 |    nan
traffic sign  | 0.651 |    nan
vegetation    | 0.904 |    nan
terrain       | 0.545 |    nan
sky           | 0.919 |    nan
person        | 0.707 |  0.498
rider         | 0.466 |  0.285
car           | 0.907 |  0.820
truck         | 0.487 |  0.186
bus           | 0.705 |  0.417
train         | 0.608 |  0.328
motorcycle    | 0.412 |  0.225
bicycle       | 0.647 |  0.442
<b>Score Average | <b>0.665 | <b>0.400

<b><u>IoU Over Categories on validation set of Cityscapes</b></u>

categories    |  IoU   |  nIoU
--------------|--------|--------
flat          | 0.965  |   nan
construction  | 0.900  |   nan
object        | 0.567  |   nan
nature        | 0.906  |   nan
sky           | 0.919  |   nan
human         | 0.732  | 0.534
vehicle       | 0.904  | 0.802
<b>Score Average | <b>0.842  | <b>0.668

 <b><u>Validation mIoU on BDD100K dataset</b></u>
 Input size    |  Parameters (million)   |  mIoU (%)
 --------------|-------------------------|-------
 768*1280      |         1.2             |   51.2
 
 <b><u>Validation and test mIoU on CamVid dataset</b></u>
 Input size    |  Validation mIoU (%)   |  Test mIoU (%)
 --------------|------------------------|-------
 640*896       |         78.6           |   71.3
