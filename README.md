# Code and data to reproduce the study of "Underwater Fish Detection using Deep Learning for Water Power Applications"

This repostitory is a fork from https://github.com/experiencor/keras-yolo3

## Model testing results:
Wells Dam fish ladder:
![Alt text](media/WellsDamPredict.gif?raw=true "Fish Detection at Wells Dam Model output")

Some true and false detections in the paper (green: human annotations; red: model detections):
![Alt text](media/fig3_detections_new.png?raw=true "Paper Figure 3")


## Dataset and Model

MHK and hydropower underwater videos and fish annotations: 

## Todo list:
- [x] Yolo3 detection
- [x] Yolo3 training (warmup and multi-scale)
- [x] mAP Evaluation
- [x] Multi-GPU training
- [x] Evaluation on VOC
- [ ] Evaluation on COCO
- [ ] MobileNet, DenseNet, ResNet, and VGG backends

## Try to detect fish in your video or image using our trained model

```python predict.py -c config_5.json -i myfishvideo.mp4``` 

The purpose of config_5.json file is to specify model settings and to load our trained model weights using the three MHK and hydropower underwater video datasets that we described earlier. 

If the detection results are good, congratulations!

If the detection results are bad, you may want to re-train the model on your dataset. To do this, follow the next steps.

## If you want to re-train the model on your fish dataset, here are the steps to follow:

### 1. Data preparation 

Data need to be input in the format of images and annotation in VOC format (.xml). You may find this script useful to generate the image and xml files: dl_utility.py

Organize the dataset into 4 folders:

+ train_image_folder <= the folder that contains the train images.

+ train_annot_folder <= the folder that contains the train annotations in VOC format.

+ valid_image_folder <= the folder that contains the validation images.

+ valid_annot_folder <= the folder that contains the validation annotations in VOC format.
    
There is a one-to-one correspondence by file name between images and annotations. If the validation set is empty, the training set will be automatically splitted into the training set and validation set using the ratio of 0.8.

### 2. Edit the configuration file
The configuration file is a json file, which looks like this:

```python
{
    "model" : {
        "min_input_size":       352,
        "max_input_size":       448,
        "anchors":              [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326],
        "labels":               ["fish"]
    },

    "train": {
        "train_image_folder":   "../train_x/",
        "train_annot_folder":   "../train_y/",      
          
        "train_times":          10,             # the number of time to cycle through the training set, useful for small datasets
        "pretrained_weights":   "",             # specify the path of the pretrained weights, but it's fine to start from scratch
        "batch_size":           20,             # the number of images to read in each batch
        "learning_rate":        1e-4,           # the base learning rate of the default Adam rate scheduler
        "nb_epoch":             100,             # number of epoches
        "warmup_epochs":        3,              # the number of initial epochs during which the sizes of the 5 boxes in each cell is forced to match the sizes of the 5 anchors, this trick seems to improve precision emperically
        "ignore_thresh":        0.5,           # noticed that this parameter does not affect evalute.py. Need to manually specify under utils.py if want to modify.
        "gpus":                 "0,1",

        "saved_weights_name":   "fish_wx_v5_all_data.h5",
        "debug":                true            # turn on/off the line that prints current confidence, position, size, class losses and recall
    },

    "valid": {
        "valid_image_folder":   "../vali_x/",
        "valid_annot_folder":   "../vali_y/",

        "valid_times":          1
    }
}

```

The ```labels``` setting lists the labels to be trained on. Only images, which has labels being listed, are fed to the network. The rest images are simply ignored. By this way, a Dog Detector can easily be trained using VOC or COCO dataset by setting ```labels``` to ```['dog']```.

### 3. Generate anchors for your dataset (optional)

`python gen_anchors.py -c config_5.json`

Copy the generated anchors printed on the terminal to the ```anchors``` setting in ```config.json```.

### 4. Start the training process

`python train.py -c config_5.json`

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 3 consecutive epoches.

### 5. Perform detection using trained weights on image, set of images, video, or webcam
`python predict.py -c config_5.json -i /path/to/image/or/video`

It carries out detection on the image and write the image with detected bounding boxes to the same folder.

## Evaluation

`python evaluate.py -c config_5.json`

This step compute the mAP performance of the model defined in `saved_weights_name` on the validation dataset defined in `valid_image_folder` and `valid_annot_folder`.
A ".csv" file will be generated after evaluation within the main directory. This csv file contains the cummulative statistics of true positives, false positives, precision and recall. The precision-recall curve in the paper are generated with ignore_thresh as 0.01, while the detections are made with ignore_thresh as 0.5.

By default the option to output red detection boxes and green ground truth boxes during evaluation is turned off. To turn them on, go to utils/utils.py and 
uncomment the two lines, and you can also customize where you want the outputs to be:
`write_predict_boxes_xml(boxes=pred_boxes, output_path='output/debug/', image_path=generator.instances[i]['filename'],  image=raw_image[0], labels=['fish'], obj_thresh=obj_thresh)`

`cv2.imwrite('output/debug/' + generator.instances[i]['filename'].split('/')[-1], np.uint8(raw_image[0]))`

