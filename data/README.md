## How to prepare data for training model (to detect your custom objects):
(to train old Yolo v2 `yolov2-voc.cfg`, `yolov2-tiny-voc.cfg`, `yolo-voc.cfg`, `yolo-voc.2.0.cfg`, ... [click by the link](https://github.com/AlexeyAB/darknet/tree/47c7af1cea5bbdedf1184963355e6418cb8b1b4f#how-to-train-pascal-voc-data))

Training Yolo v4:

0. For training `cfg/yolov4-custom.cfg` download the pre-trained weights-file (162 MB) in directory `darknet`: [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) (Google drive mirror [yolov4.conv.137](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp) )

1. Create file `yolo-obj.cfg` with the same content as in `yolov4-custom.cfg` (or copy `yolov4-custom.cfg` to `yolo-obj.cfg)` and:

  * change line batch to [`batch=64`](https://github.com/BTTHuyen/YOLOv4/blob/c5dae763ca31e35ba8735ebabcf9105fffd747e0/cfg/yolov4-custom.cfg#L6)
  * change line subdivisions to [`subdivisions=16`](https://github.com/BTTHuyen/YOLOv4/blob/c5dae763ca31e35ba8735ebabcf9105fffd747e0/cfg/yolov4-custom.cfg#L7)
  * change line max_batches to (`classes*2000` but not less than number of training images, but not less than number of training images and not less than `6000`), f.e. [`max_batches=6000`](https://github.com/BTTHuyen/YOLOv4/blob/c5dae763ca31e35ba8735ebabcf9105fffd747e0/cfg/yolov4-custom.cfg#L20) if you train for 3 classes
  * change line steps to 80% and 90% of max_batches, f.e. [`steps=4800,5400`](https://github.com/BTTHuyen/YOLOv4/blob/c5dae763ca31e35ba8735ebabcf9105fffd747e0/cfg/yolov4-custom.cfg#L22)
  * set network size `width=416 height=416` or any value multiple of 32 at line 8,9 : https://github.com/BTTHuyen/YOLOv4/blob/c5dae763ca31e35ba8735ebabcf9105fffd747e0/cfg/yolov4-custom.cfg#L8-L9
  * change line `classes=80` to your number of objects in each of 3 `[yolo]`-layers at line 970, 1058, 1146 :
      * https://github.com/AlexeyAB/darknet/blob/8f900493c634eb83a895776b2a130969ff9c1ec5/cfg/yolov4-custom.cfg#L970
      * https://github.com/AlexeyAB/darknet/blob/8f900493c634eb83a895776b2a130969ff9c1ec5/cfg/yolov4-custom.cfg#L1058
      * https://github.com/AlexeyAB/darknet/blob/8f900493c634eb83a895776b2a130969ff9c1ec5/cfg/yolov4-custom.cfg#L1146
  * change [`filters=255`] to filters=(classes + 5)x3 in the 3 `[convolutional]` before each `[yolo]` layer, keep in mind that it only has to be the last `[convolutional]` before each of the `[yolo]` layers at line 963, 1051, 1139.
      * https://github.com/AlexeyAB/darknet/blob/8f900493c634eb83a895776b2a130969ff9c1ec5/cfg/yolov4-custom.cfg#L963
      * https://github.com/AlexeyAB/darknet/blob/8f900493c634eb83a895776b2a130969ff9c1ec5/cfg/yolov4-custom.cfg#L1051
      * https://github.com/AlexeyAB/darknet/blob/8f900493c634eb83a895776b2a130969ff9c1ec5/cfg/yolov4-custom.cfg#L1139
  * when using [`[Gaussian_yolo]`](https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L608)  layers, change [`filters=57`] filters=(classes + 9)x3 in the 3 `[convolutional]` before each `[Gaussian_yolo]` layer (training YOLO v3)
      * https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L604
      * https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L696
      * https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L789
      
  So if `classes=1` then should be `filters=18`. If `classes=2` then write `filters=21`.
  
  **(Do not write in the cfg-file: filters=(classes + 5)x3)**
  
  (Generally `filters` depends on the `classes`, `coords` and number of `mask`s, i.e. filters=`(classes + coords + 1)*<number of mask>`, where `mask` is indices of anchors. If `mask` is absence, then filters=`(classes + coords + 1)*num`)

  So for example, for 2 objects, your file `yolo-obj.cfg` should differ from `yolov4-custom.cfg` in such lines in each of **3** [yolo]-layers:

  ```
  [convolutional]
  filters=21

  [region]
  classes=2
  ```
  
  Please visit this website: https://github.com/BTTHuyen/YOLOv4/wiki/Config-parameters to see more detail about config parameters

2. Create file `obj.names` in the directory `darknet/config`, with objects names - each in new line

```
Object_name 1
Object_name 2
```

3. Create file `obj.data` in the directory `darknet/config`, containing (where **classes = number of objects**):

  ```
  classes= 2
  train  = data/train.txt
  valid  = data/test.txt
  names = cfg/obj.names
  backup = backup/
  ```

4. Put image-files (.jpg) of your objects in the directory `darknet\data\obj\`

5. You should label each object on images from your dataset. Use this visual GUI-software for marking bounded boxes of objects and generating annotation files 

It will create `.txt`-file for each `.jpg`-image-file - in the same directory and with the same name, but with `.txt`-extension, and put to file: object number and object coordinates on this image, for each object in new line: 

`<object-class> <x_center> <y_center> <width> <height>`

  Where: 
  * `<object-class>` - integer object number from `0` to `(classes-1)`
  * `<x_center> <y_center> <width> <height>` - float values **relative** to width and height of image, it can be equal from `(0.0 to 1.0]`
  * for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
  * atention: `<x_center> <y_center>` - are center of rectangle (are not top-left corner)

  For example for `img1.jpg` you will be created `img1.txt` containing:

  ```
  1 0.716797 0.395833 0.216406 0.147222
  0 0.687109 0.379167 0.255469 0.158333
  1 0.420312 0.395833 0.140625 0.166667
  ```

6. Create file `train.txt, test.txt` in directory `darknet\data\`, with filenames of your images, each filename in new line. You can use [Make_train_Test.py](https://github.com/BTTHuyen/YOLOv4/blob/master/Make_Train_Test.py) to create 2 files. for example containing:

  ```
  data/obj/img1.jpg
  data/obj/img2.jpg
  data/obj/img3.jpg
  ```

## How to mark bounded boxes of objects and create annotation files:
Here you can find repository with GUI-software for marking bounded boxes of objects and generating annotation files.

Different tools for marking objects in images:
1. in C++: https://github.com/AlexeyAB/Yolo_mark 
2. in Python: https://github.com/tzutalin/labelImg
3. in Python: https://github.com/Cartucho/OpenLabeling
4. in C++: https://www.ccoderun.ca/darkmark/
5. in JavaScript: https://github.com/opencv/cvat