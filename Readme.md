# Using Network slimming 

## Params information

+ Layers : 
  + if Renset 101 -> Layers = [3,4,23,3]
+ cfg :
  + target pruning Conv list
  + cfg[0] -> not bottleneck structure
  + When making Mask-R CNN model using Pytorch package, FPN needs cfg information
  + also it is needed for reconstructing pruned Resnet network

## Result

+ Because COCO images are large scale and I don't have super GPU, I couldn't train Mask R-CNN by e2e.

## Flow

+ Train -> get pruning model -> fine-tuning or retraining from scratch

## Dataset

+ you need to download coco dataset for year 2017
+ directory structure
  + coco_dataset
    + annotation
    + train2017
    + val2017

## Training

+ See model_train/light_mask_rcnn

+ ```
  if __name__ == "__main__":
      # the number of bottlenet layers for Resnet 101
      layers = [3, 4, 23, 3]
      root="path_to_your_coco_dataset"
      #num_classes: the number of classes
      model = get_mask_rcnn_model(layers=layers,num_classes=91)
      train_loader = get_coco_dataloader(root_path=root, batch=1,Train=True)
      val_loader = get_coco_dataloader(root_path=root, batch=1, Train=False)
      train(model=model,epochs=40,train_loader=train_loader,val_loader=val_loader)
  
      #get pruned structure and train again from scratch or fine-tuning
      # but if you want to fine-tune, you have to add the codes for copying the weight of trained conv layers in get_pruned_config function
      pruned_cfg = get_pruned_config(model,0.4)
      model = get_mask_rcnn_model(layers,num_classes=91,cfg=pruned_cfg)
      train(model=model, epochs=40, train_loader=train_loader, val_loader=val_loader)
  ```

## requirement 

+ pycocotools
  + pip3 install Cython
  + window :
    + pip3 install numpy==1.17.4
    + pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
  + linux:
    + git clone https://github.com/cocodataset/cocoapi.git
    + cd PythonAPI
    + make
  
+ pytorch version >=1.4

  

  

## Reference

+ https://github.com/Eric-mingjie/network-slimming.git
+ Learning Efficient Convolutional Networks through Network Slimming
+ RETHINKING THE VALUE OF NETWORK PRUNING

