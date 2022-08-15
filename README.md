# Projeto Final - Modelos Preditivos Conexionistas

### ROBERTO SÁ BARRETO PAIVA DA CUNHA

|**Classificação de imagens**|**YOLOV5**|**PYTHON**|
|--|--|--|
|Classificação de Imagens<br>ou<br>Deteção de Objetos| YOLOv5

## Performance

O modelo treinado possui performance de **97.93%**.

### Output do bloco de treinamento

<details>
  <summary>Click to expand!</summary>
  
  ```
  train: weights=yolov5s.pt, cfg=, data=/content/datasets/Projeto-Final---Modelos-Preditivos-Conexionistas-1/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=150, batch_size=16, imgsz=416, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v6.1-394-gd7bc5d7 Python-3.7.13 torch-1.12.1+cu113 CUDA:0 (Tesla T4, 15110MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Weights & Biases: run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs in Weights & Biases
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 runs in ClearML
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
100% 755k/755k [00:00<00:00, 20.1MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt to yolov5s.pt...
100% 14.1M/14.1M [00:00<00:00, 39.8MB/s]

Overriding model.yaml nc=80 with nc=3

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     21576  models.yolo.Detect                      [3, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model summary: 270 layers, 7027720 parameters, 7027720 gradients, 16.0 GFLOPs

Transferred 343/349 items from yolov5s.pt
AMP: checks passed ✅
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
albumentations: Blur(always_apply=False, p=0.01, blur_limit=(3, 7)), MedianBlur(always_apply=False, p=0.01, blur_limit=(3, 7)), ToGray(always_apply=False, p=0.01), CLAHE(always_apply=False, p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning '/content/datasets/Projeto-Final---Modelos-Preditivos-Conexionistas-1/train/labels' images and labels...63 found, 0 missing, 0 empty, 0 corrupt: 100% 63/63 [00:00<00:00, 2140.84it/s]
train: New cache created: /content/datasets/Projeto-Final---Modelos-Preditivos-Conexionistas-1/train/labels.cache
train: Caching images (0.0GB ram): 100% 63/63 [00:00<00:00, 391.20it/s]
val: Scanning '/content/datasets/Projeto-Final---Modelos-Preditivos-Conexionistas-1/valid/labels' images and labels...18 found, 0 missing, 0 empty, 0 corrupt: 100% 18/18 [00:00<00:00, 705.58it/s]
val: New cache created: /content/datasets/Projeto-Final---Modelos-Preditivos-Conexionistas-1/valid/labels.cache
val: Caching images (0.0GB ram): 100% 18/18 [00:00<00:00, 120.93it/s]
Plotting labels to runs/train/exp/labels.jpg... 

AutoAnchor: 2.35 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Image sizes 416 train, 416 val
Using 2 dataloader workers
Logging results to runs/train/exp
Starting training for 150 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     0/149     1.71G     0.107    0.0189   0.03766        28       416: 100% 4/4 [00:04<00:00,  1.16s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:01<00:00,  1.61s/it]
                 all         18         18    0.00334          1     0.0275     0.0167

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     1/149     1.71G     0.103   0.01622   0.03581        19       416: 100% 4/4 [00:00<00:00,  6.00it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.68it/s]
                 all         18         18    0.00333          1     0.0288     0.0157

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     2/149     1.71G   0.08499   0.01812   0.03538        31       416: 100% 4/4 [00:00<00:00,  5.86it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.99it/s]
                 all         18         18    0.00334          1     0.0224     0.0114

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     3/149     1.71G   0.08833   0.02058   0.03595        34       416: 100% 4/4 [00:00<00:00,  5.80it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  5.70it/s]
                 all         18         18    0.00334          1     0.0261     0.0118

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     4/149     1.71G   0.09365    0.0208   0.03692        38       416: 100% 4/4 [00:00<00:00,  6.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  5.74it/s]
                 all         18         18    0.00334          1     0.0437     0.0269

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     5/149     1.71G    0.0988   0.02059    0.0338        27       416: 100% 4/4 [00:00<00:00,  6.32it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  6.03it/s]
                 all         18         18    0.00333          1      0.072     0.0314

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     6/149     1.71G   0.08068   0.02065   0.03539        39       416: 100% 4/4 [00:00<00:00,  6.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  6.17it/s]
                 all         18         18    0.00333          1      0.104     0.0327

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     7/149     1.71G   0.06953   0.01985   0.03518        40       416: 100% 4/4 [00:00<00:00,  6.63it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  6.63it/s]
                 all         18         18    0.00334          1      0.142      0.048

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     8/149     1.71G   0.07111   0.01966   0.03229        32       416: 100% 4/4 [00:00<00:00,  6.64it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  6.89it/s]
                 all         18         18    0.00334          1       0.14     0.0727

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     9/149     1.71G   0.07193   0.01891   0.03318        29       416: 100% 4/4 [00:00<00:00,  6.34it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  6.96it/s]
                 all         18         18    0.00387          1     0.0964     0.0346

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    10/149     1.71G   0.05719   0.02108   0.02575        29       416: 100% 4/4 [00:00<00:00,  6.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  6.86it/s]
                 all         18         18     0.0298      0.833      0.193     0.0493

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    11/149     1.71G     0.062   0.02077   0.02689        28       416: 100% 4/4 [00:00<00:00,  6.37it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.55it/s]
                 all         18         18     0.0768        0.5      0.268      0.144

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    12/149     1.71G   0.06577   0.01897   0.03407        23       416: 100% 4/4 [00:00<00:00,  6.37it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.80it/s]
                 all         18         18      0.103      0.611      0.194     0.0722

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    13/149     1.71G   0.06511   0.02197   0.03187        33       416: 100% 4/4 [00:00<00:00,  6.31it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.78it/s]
                 all         18         18      0.203      0.667      0.322      0.127

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    14/149     1.71G   0.05815   0.02131   0.03306        36       416: 100% 4/4 [00:00<00:00,  6.34it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.60it/s]
                 all         18         18      0.532        0.5      0.316      0.172

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    15/149     1.71G   0.06745   0.02173   0.03477        37       416: 100% 4/4 [00:00<00:00,  6.63it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.81it/s]
                 all         18         18      0.554      0.389       0.39      0.197

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    16/149     1.71G   0.05258   0.01899   0.03318        34       416: 100% 4/4 [00:00<00:00,  7.01it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.41it/s]
                 all         18         18      0.553      0.389      0.391      0.167

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    17/149     1.71G   0.05297   0.01913   0.03222        35       416: 100% 4/4 [00:00<00:00,  6.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.61it/s]
                 all         18         18       0.59      0.524      0.395      0.203

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    18/149     1.71G   0.06854   0.01926   0.03269        36       416: 100% 4/4 [00:00<00:00,  6.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.35it/s]
                 all         18         18      0.582        0.5      0.381      0.247

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    19/149     1.71G   0.04479   0.01645   0.02617        35       416: 100% 4/4 [00:00<00:00,  6.82it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.13it/s]
                 all         18         18      0.557      0.556       0.39      0.207

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    20/149     1.71G   0.05452   0.01845   0.03255        40       416: 100% 4/4 [00:00<00:00,  6.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.68it/s]
                 all         18         18      0.618      0.608      0.444      0.294

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    21/149     1.71G   0.04974   0.01724   0.03206        28       416: 100% 4/4 [00:00<00:00,  6.34it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.55it/s]
                 all         18         18      0.559      0.611      0.409      0.242

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    22/149     1.71G    0.0543   0.01753   0.03121        26       416: 100% 4/4 [00:00<00:00,  6.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.34it/s]
                 all         18         18      0.495        0.5      0.441      0.266

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    23/149     1.71G   0.04017    0.0187   0.02906        21       416: 100% 4/4 [00:00<00:00,  6.74it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.38it/s]
                 all         18         18      0.398      0.736      0.323      0.182

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    24/149     1.71G   0.06078   0.01643   0.03038        34       416: 100% 4/4 [00:00<00:00,  6.68it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.69it/s]
                 all         18         18        0.5      0.259      0.216      0.115

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    25/149     1.71G   0.05103   0.01442   0.02528        25       416: 100% 4/4 [00:00<00:00,  6.77it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.65it/s]
                 all         18         18      0.581      0.222      0.267      0.118

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    26/149     1.71G   0.04128   0.01659   0.02463        28       416: 100% 4/4 [00:00<00:00,  6.87it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.41it/s]
                 all         18         18      0.606      0.327      0.346      0.231

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    27/149     1.71G    0.0547   0.01618   0.02607        31       416: 100% 4/4 [00:00<00:00,  6.46it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.93it/s]
                 all         18         18       0.58      0.222      0.226      0.121

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    28/149     1.71G   0.05391   0.01814   0.02378        34       416: 100% 4/4 [00:00<00:00,  6.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.06it/s]
                 all         18         18      0.483      0.333      0.363      0.182

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    29/149     1.71G   0.05311   0.01591   0.02786        37       416: 100% 4/4 [00:00<00:00,  6.35it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.16it/s]
                 all         18         18       0.43      0.333      0.305      0.128

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    30/149     1.71G   0.04071   0.01628   0.02503        24       416: 100% 4/4 [00:00<00:00,  6.75it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.87it/s]
                 all         18         18      0.582      0.333      0.277      0.105

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    31/149     1.71G   0.04176   0.01711   0.02264        38       416: 100% 4/4 [00:00<00:00,  6.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.25it/s]
                 all         18         18       0.52      0.395      0.357      0.202

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    32/149     1.71G   0.04615   0.01843   0.02477        30       416: 100% 4/4 [00:00<00:00,  6.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.48it/s]
                 all         18         18      0.593      0.333      0.434      0.212

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    33/149     1.71G   0.05256    0.0174   0.03088        31       416: 100% 4/4 [00:00<00:00,  6.41it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.68it/s]
                 all         18         18      0.469        0.5      0.489      0.187

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    34/149     1.71G   0.05454   0.01593   0.02757        34       416: 100% 4/4 [00:00<00:00,  6.31it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.04it/s]
                 all         18         18      0.229        0.5      0.359      0.109

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    35/149     1.71G   0.04837   0.01572    0.0229        27       416: 100% 4/4 [00:00<00:00,  6.62it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.53it/s]
                 all         18         18      0.608        0.5      0.576      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    36/149     1.71G   0.04776   0.01664   0.02552        36       416: 100% 4/4 [00:00<00:00,  6.76it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.10it/s]
                 all         18         18      0.467      0.609      0.576      0.312

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    37/149     1.71G   0.03922   0.01774   0.02976        30       416: 100% 4/4 [00:00<00:00,  6.15it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  6.49it/s]
                 all         18         18      0.702      0.333      0.577      0.269

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    38/149     1.71G   0.03791   0.01607   0.02733        30       416: 100% 4/4 [00:00<00:00,  6.30it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.12it/s]
                 all         18         18      0.246      0.644      0.411      0.285

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    39/149     1.71G   0.04781   0.01639   0.02803        40       416: 100% 4/4 [00:00<00:00,  6.50it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.97it/s]
                 all         18         18      0.298      0.632      0.531      0.284

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    40/149     1.71G   0.04577   0.01816   0.02784        29       416: 100% 4/4 [00:00<00:00,  6.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.96it/s]
                 all         18         18      0.686      0.357       0.45      0.221

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    41/149     1.71G   0.03763   0.01509    0.0197        29       416: 100% 4/4 [00:00<00:00,  6.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.44it/s]
                 all         18         18      0.314      0.563      0.539      0.363

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    42/149     1.71G   0.05028   0.01597   0.02807        28       416: 100% 4/4 [00:00<00:00,  6.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.60it/s]
                 all         18         18      0.768        0.5      0.603      0.356

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    43/149     1.71G   0.05157   0.01597   0.02141        27       416: 100% 4/4 [00:00<00:00,  6.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.56it/s]
                 all         18         18      0.419      0.556      0.653      0.372

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    44/149     1.71G   0.04404    0.0161   0.02747        31       416: 100% 4/4 [00:00<00:00,  6.77it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.36it/s]
                 all         18         18       0.74      0.496       0.54      0.341

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    45/149     1.71G   0.04722   0.01357   0.02222        23       416: 100% 4/4 [00:00<00:00,  6.46it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.43it/s]
                 all         18         18      0.477      0.551      0.653      0.426

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    46/149     1.71G   0.04491   0.01628   0.02261        40       416: 100% 4/4 [00:00<00:00,  6.89it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.86it/s]
                 all         18         18        0.4      0.444      0.552      0.362

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    47/149     1.71G   0.04004   0.01516   0.02921        25       416: 100% 4/4 [00:00<00:00,  6.58it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.22it/s]
                 all         18         18      0.458      0.619      0.681      0.437

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    48/149     1.71G   0.03815   0.01551   0.02593        32       416: 100% 4/4 [00:00<00:00,  6.18it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.53it/s]
                 all         18         18      0.822      0.527      0.695      0.422

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    49/149     1.71G   0.03079   0.01567   0.02138        33       416: 100% 4/4 [00:00<00:00,  6.71it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.45it/s]
                 all         18         18      0.381       0.53      0.605      0.377

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    50/149     1.71G   0.05439   0.01593   0.02989        39       416: 100% 4/4 [00:00<00:00,  6.66it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.58it/s]
                 all         18         18      0.377      0.667      0.603      0.345

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    51/149     1.71G   0.03536   0.01712    0.0229        36       416: 100% 4/4 [00:00<00:00,  6.62it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.90it/s]
                 all         18         18      0.344      0.611      0.563       0.39

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    52/149     1.71G   0.02649   0.01537   0.02017        28       416: 100% 4/4 [00:00<00:00,  6.93it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.62it/s]
                 all         18         18      0.337      0.674      0.495      0.288

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    53/149     1.71G   0.02716    0.0142   0.01879        34       416: 100% 4/4 [00:00<00:00,  6.38it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.39it/s]
                 all         18         18      0.481      0.722      0.659      0.412

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    54/149     1.71G   0.04271   0.01628    0.0259        35       416: 100% 4/4 [00:00<00:00,  6.74it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.19it/s]
                 all         18         18       0.81        0.5      0.614      0.436

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    55/149     1.71G   0.03832   0.01423   0.02453        37       416: 100% 4/4 [00:00<00:00,  6.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.76it/s]
                 all         18         18      0.498      0.685      0.692      0.464

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    56/149     1.71G     0.031   0.01548   0.02264        29       416: 100% 4/4 [00:00<00:00,  6.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.58it/s]
                 all         18         18      0.472      0.667      0.696      0.509

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    57/149     1.71G   0.03438   0.01537   0.01997        21       416: 100% 4/4 [00:00<00:00,  6.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.85it/s]
                 all         18         18      0.526      0.722      0.682      0.491

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    58/149     1.71G    0.0474    0.0171   0.02396        47       416: 100% 4/4 [00:00<00:00,  5.92it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.81it/s]
                 all         18         18      0.412      0.722      0.671      0.533

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    59/149     1.71G   0.03934   0.01654   0.02368        36       416: 100% 4/4 [00:00<00:00,  6.65it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.79it/s]
                 all         18         18      0.454      0.778      0.723      0.557

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    60/149     1.71G   0.04338    0.0172   0.02332        33       416: 100% 4/4 [00:00<00:00,  6.41it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.62it/s]
                 all         18         18       0.94      0.556      0.737      0.508

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    61/149     1.71G   0.03379   0.01486   0.01793        30       416: 100% 4/4 [00:00<00:00,  6.67it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.18it/s]
                 all         18         18      0.794      0.563      0.694      0.488

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    62/149     1.71G   0.03604   0.01309   0.02016        29       416: 100% 4/4 [00:00<00:00,  6.10it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.70it/s]
                 all         18         18      0.846      0.556      0.725      0.493

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    63/149     1.71G   0.03682   0.01758   0.02426        33       416: 100% 4/4 [00:00<00:00,  6.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.43it/s]
                 all         18         18      0.533      0.611      0.742      0.441

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    64/149     1.71G   0.03018   0.01353   0.02442        30       416: 100% 4/4 [00:00<00:00,  6.42it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.97it/s]
                 all         18         18      0.898      0.523      0.688      0.462

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    65/149     1.71G   0.03487   0.01534   0.03021        39       416: 100% 4/4 [00:00<00:00,  6.54it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.36it/s]
                 all         18         18       0.89        0.5       0.69      0.454

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    66/149     1.71G   0.02096   0.01408   0.01647        32       416: 100% 4/4 [00:00<00:00,  6.75it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.90it/s]
                 all         18         18      0.882      0.534      0.765      0.518

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    67/149     1.71G   0.03456   0.01493   0.01813        35       416: 100% 4/4 [00:00<00:00,  6.11it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.59it/s]
                 all         18         18      0.494      0.764      0.774      0.511

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    68/149     1.71G   0.02842   0.01416   0.02775        25       416: 100% 4/4 [00:00<00:00,  6.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.59it/s]
                 all         18         18      0.508      0.795      0.771      0.491

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    69/149     1.71G   0.03607    0.0152   0.02855        36       416: 100% 4/4 [00:00<00:00,  6.66it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.78it/s]
                 all         18         18      0.494      0.796      0.768      0.511

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    70/149     1.71G   0.02575   0.01423    0.0182        28       416: 100% 4/4 [00:00<00:00,  6.63it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.47it/s]
                 all         18         18      0.418      0.823      0.658      0.455

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    71/149     1.71G   0.03317   0.01598   0.02613        25       416: 100% 4/4 [00:00<00:00,  6.65it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.21it/s]
                 all         18         18      0.507      0.881      0.777      0.487

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    72/149     1.71G   0.02465   0.01501   0.01977        34       416: 100% 4/4 [00:00<00:00,  6.28it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.82it/s]
                 all         18         18      0.442      0.889      0.734      0.457

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    73/149     1.71G   0.03637   0.01329   0.04081        37       416: 100% 4/4 [00:00<00:00,  6.82it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.02it/s]
                 all         18         18      0.526      0.882      0.825      0.506

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    74/149     1.71G    0.0307   0.01277   0.01967        29       416: 100% 4/4 [00:00<00:00,  7.06it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.03it/s]
                 all         18         18      0.354      0.833      0.624       0.43

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    75/149     1.71G   0.03523    0.0144   0.01913        35       416: 100% 4/4 [00:00<00:00,  6.50it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.09it/s]
                 all         18         18      0.382      0.889      0.656      0.488

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    76/149     1.71G   0.04191   0.01393   0.02434        35       416: 100% 4/4 [00:00<00:00,  6.85it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.59it/s]
                 all         18         18      0.448      0.794      0.747      0.568

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    77/149     1.71G   0.02355   0.01217   0.01789        34       416: 100% 4/4 [00:00<00:00,  6.94it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.13it/s]
                 all         18         18      0.457      0.889      0.766      0.536

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    78/149     1.71G   0.03164   0.01325   0.02914        36       416: 100% 4/4 [00:00<00:00,  6.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.63it/s]
                 all         18         18      0.473      0.778      0.793      0.581

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    79/149     1.71G   0.02531   0.01486   0.02337        37       416: 100% 4/4 [00:00<00:00,  6.47it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.45it/s]
                 all         18         18      0.421      0.779      0.776      0.537

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    80/149     1.71G   0.03933    0.0146   0.02077        31       416: 100% 4/4 [00:00<00:00,  6.29it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.28it/s]
                 all         18         18      0.464       0.82      0.795       0.62

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    81/149     1.71G   0.02831   0.01454   0.01912        27       416: 100% 4/4 [00:00<00:00,  6.31it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.46it/s]
                 all         18         18      0.496      0.841      0.832      0.612

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    82/149     1.71G   0.03001   0.01444   0.02112        32       416: 100% 4/4 [00:00<00:00,  6.63it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.86it/s]
                 all         18         18      0.501      0.832      0.832      0.597

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    83/149     1.71G   0.02706   0.01322   0.02445        27       416: 100% 4/4 [00:00<00:00,  6.78it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.28it/s]
                 all         18         18       0.52      0.923      0.793      0.574

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    84/149     1.71G   0.03602   0.01477    0.0272        33       416: 100% 4/4 [00:00<00:00,  6.69it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.64it/s]
                 all         18         18      0.495      0.835      0.805      0.536

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    85/149     1.71G   0.02224   0.01265   0.02168        24       416: 100% 4/4 [00:00<00:00,  6.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.10it/s]
                 all         18         18      0.515      0.913      0.813      0.568

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    86/149     1.71G   0.03182   0.01452   0.02306        26       416: 100% 4/4 [00:00<00:00,  6.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.87it/s]
                 all         18         18      0.528      0.885      0.801      0.574

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    87/149     1.71G   0.01914   0.01278    0.0142        27       416: 100% 4/4 [00:00<00:00,  6.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.17it/s]
                 all         18         18      0.588      0.864       0.87      0.547

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    88/149     1.71G   0.02864   0.01269    0.0175        26       416: 100% 4/4 [00:00<00:00,  6.67it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.40it/s]
                 all         18         18      0.569      0.829      0.864       0.61

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    89/149     1.71G   0.02922     0.014   0.02165        34       416: 100% 4/4 [00:00<00:00,  6.63it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.38it/s]
                 all         18         18      0.563      0.807      0.871       0.62

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    90/149     1.71G    0.0307   0.01493   0.02039        38       416: 100% 4/4 [00:00<00:00,  6.63it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.82it/s]
                 all         18         18      0.567      0.822      0.874      0.601

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    91/149     1.71G   0.03316   0.01277   0.02621        36       416: 100% 4/4 [00:00<00:00,  6.15it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.75it/s]
                 all         18         18      0.675      0.831      0.916      0.588

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    92/149     1.71G   0.02832   0.01248   0.02573        33       416: 100% 4/4 [00:00<00:00,  6.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.46it/s]
                 all         18         18      0.616      0.911      0.923      0.598

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    93/149     1.71G   0.02417   0.01436   0.02002        36       416: 100% 4/4 [00:00<00:00,  6.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.88it/s]
                 all         18         18      0.618      0.852      0.871      0.524

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    94/149     1.71G   0.02887   0.01336    0.0206        28       416: 100% 4/4 [00:00<00:00,  6.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.75it/s]
                 all         18         18      0.659      0.897      0.944       0.63

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    95/149     1.71G   0.02222   0.01327    0.0162        35       416: 100% 4/4 [00:00<00:00,  6.23it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.06it/s]
                 all         18         18      0.698      0.854      0.944      0.608

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    96/149     1.71G   0.02965   0.01616   0.01982        27       416: 100% 4/4 [00:00<00:00,  6.46it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.78it/s]
                 all         18         18      0.719      0.802      0.918      0.662

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    97/149     1.71G   0.03166   0.01479   0.02835        31       416: 100% 4/4 [00:00<00:00,  6.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.98it/s]
                 all         18         18      0.733      0.824      0.934      0.626

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    98/149     1.71G   0.02879   0.01328   0.01993        26       416: 100% 4/4 [00:00<00:00,  6.19it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.95it/s]
                 all         18         18       0.71      0.882      0.923      0.633

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    99/149     1.71G   0.02484   0.01433   0.01641        35       416: 100% 4/4 [00:00<00:00,  6.40it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.39it/s]
                 all         18         18      0.692       0.86      0.869       0.59

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   100/149     1.71G   0.03424   0.01375   0.01645        34       416: 100% 4/4 [00:00<00:00,  6.23it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.53it/s]
                 all         18         18      0.708      0.909      0.914      0.578

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   101/149     1.71G   0.03046   0.01535   0.01894        32       416: 100% 4/4 [00:00<00:00,  6.27it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.24it/s]
                 all         18         18       0.59      0.832      0.827      0.563

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   102/149     1.71G   0.02362   0.01541   0.01985        34       416: 100% 4/4 [00:00<00:00,  6.66it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.64it/s]
                 all         18         18      0.577      0.829      0.787      0.484

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   103/149     1.71G   0.02928   0.01268   0.02285        31       416: 100% 4/4 [00:00<00:00,  6.65it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.97it/s]
                 all         18         18      0.606       0.86      0.826      0.494

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   104/149     1.71G   0.02364   0.01209   0.01917        32       416: 100% 4/4 [00:00<00:00,  6.46it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.44it/s]
                 all         18         18      0.624      0.839      0.819      0.502

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   105/149     1.71G   0.02973   0.01424   0.02349        28       416: 100% 4/4 [00:00<00:00,  6.51it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.08it/s]
                 all         18         18      0.795      0.918      0.901       0.51

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   106/149     1.71G   0.03711   0.01274   0.02202        26       416: 100% 4/4 [00:00<00:00,  6.06it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.03it/s]
                 all         18         18      0.853      0.885      0.902      0.594

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   107/149     1.71G   0.03327   0.01326   0.02338        34       416: 100% 4/4 [00:00<00:00,  4.32it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  6.01it/s]
                 all         18         18      0.757      0.854      0.929      0.656

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   108/149     1.71G   0.02265   0.01451   0.01421        40       416: 100% 4/4 [00:00<00:00,  4.06it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.64it/s]
                 all         18         18      0.767      0.889      0.928       0.65

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   109/149     1.71G   0.02917   0.01498   0.01942        28       416: 100% 4/4 [00:00<00:00,  6.45it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.04it/s]
                 all         18         18      0.862      0.833      0.922      0.599

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   110/149     1.71G   0.02123   0.01485   0.01756        24       416: 100% 4/4 [00:00<00:00,  5.98it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.77it/s]
                 all         18         18      0.868          1      0.987      0.588

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   111/149     1.71G   0.03132   0.01364    0.0191        40       416: 100% 4/4 [00:00<00:00,  6.25it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.23it/s]
                 all         18         18      0.782      0.778      0.841      0.467

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   112/149     1.71G   0.02474   0.01363   0.02289        33       416: 100% 4/4 [00:00<00:00,  6.73it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.53it/s]
                 all         18         18      0.784      0.733      0.832      0.485

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   113/149     1.71G   0.02673   0.01356   0.01465        31       416: 100% 4/4 [00:00<00:00,  6.23it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.49it/s]
                 all         18         18      0.817      0.833       0.84       0.54

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   114/149     1.71G   0.02764   0.01341   0.01722        40       416: 100% 4/4 [00:00<00:00,  6.37it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.85it/s]
                 all         18         18      0.834      0.833      0.854      0.504

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   115/149     1.71G     0.027   0.01398   0.02003        26       416: 100% 4/4 [00:00<00:00,  6.10it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.50it/s]
                 all         18         18      0.861      0.857      0.882      0.561

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   116/149     1.71G   0.02256   0.01219   0.01738        27       416: 100% 4/4 [00:00<00:00,  6.37it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.15it/s]
                 all         18         18      0.849       0.85       0.88      0.554

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   117/149     1.71G   0.02063   0.01248   0.01637        26       416: 100% 4/4 [00:00<00:00,  6.42it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.52it/s]
                 all         18         18        0.9      0.827      0.852      0.539

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   118/149     1.71G   0.02259   0.01378   0.01613        29       416: 100% 4/4 [00:00<00:00,  6.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.02it/s]
                 all         18         18      0.891      0.889      0.881      0.541

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   119/149     1.71G   0.02984   0.01312   0.02917        24       416: 100% 4/4 [00:00<00:00,  6.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.14it/s]
                 all         18         18      0.777      0.833      0.847      0.533

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   120/149     1.71G   0.02479   0.01434   0.01678        28       416: 100% 4/4 [00:00<00:00,  5.94it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.03it/s]
                 all         18         18      0.833      0.833       0.86      0.539

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   121/149     1.71G   0.02761   0.01329   0.01799        29       416: 100% 4/4 [00:00<00:00,  6.76it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.86it/s]
                 all         18         18      0.857      0.833      0.873      0.517

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   122/149     1.71G   0.02589   0.01344   0.02041        32       416: 100% 4/4 [00:00<00:00,  6.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.16it/s]
                 all         18         18      0.849      0.825      0.867      0.567

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   123/149     1.71G   0.02429   0.01222   0.01743        33       416: 100% 4/4 [00:00<00:00,  6.16it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.92it/s]
                 all         18         18       0.85      0.833      0.883      0.556

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   124/149     1.71G   0.01825   0.01365   0.01534        34       416: 100% 4/4 [00:00<00:00,  6.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.94it/s]
                 all         18         18      0.862      0.847      0.901      0.587

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   125/149     1.71G   0.01773   0.01304   0.01043        35       416: 100% 4/4 [00:00<00:00,  6.49it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.92it/s]
                 all         18         18      0.917      0.896       0.98      0.607

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   126/149     1.71G   0.02195   0.01304   0.01523        26       416: 100% 4/4 [00:00<00:00,  6.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.58it/s]
                 all         18         18      0.949      0.944      0.987      0.624

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   127/149     1.71G   0.01903   0.01259   0.01325        21       416: 100% 4/4 [00:00<00:00,  6.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.91it/s]
                 all         18         18      0.927      0.889       0.98      0.596

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   128/149     1.71G   0.02324   0.01262    0.0201        26       416: 100% 4/4 [00:00<00:00,  6.19it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.02it/s]
                 all         18         18      0.949      0.931      0.987      0.619

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   129/149     1.71G   0.02409   0.01249    0.0153        33       416: 100% 4/4 [00:00<00:00,  6.50it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.09it/s]
                 all         18         18      0.905      0.889       0.98      0.641

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   130/149     1.71G    0.0272    0.0144   0.02077        41       416: 100% 4/4 [00:00<00:00,  6.40it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.06it/s]
                 all         18         18      0.931      0.932      0.987      0.666

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   131/149     1.71G   0.02825   0.01296   0.02682        32       416: 100% 4/4 [00:00<00:00,  6.45it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.05it/s]
                 all         18         18      0.915      0.948      0.987      0.655

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   132/149     1.71G   0.02089   0.01315   0.01927        35       416: 100% 4/4 [00:00<00:00,  6.51it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.45it/s]
                 all         18         18      0.903      0.948      0.987      0.656

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   133/149     1.71G   0.02453   0.01353   0.01714        34       416: 100% 4/4 [00:00<00:00,  6.86it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.33it/s]
                 all         18         18      0.945      0.944      0.987      0.638

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   134/149     1.71G   0.02213   0.01274    0.0182        33       416: 100% 4/4 [00:00<00:00,  6.87it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.92it/s]
                 all         18         18      0.953      0.937      0.987      0.652

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   135/149     1.71G    0.0155     0.011   0.01429        24       416: 100% 4/4 [00:00<00:00,  6.69it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.13it/s]
                 all         18         18      0.965      0.928      0.987      0.638

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   136/149     1.71G   0.02751   0.01304   0.01953        31       416: 100% 4/4 [00:00<00:00,  6.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.15it/s]
                 all         18         18      0.969      0.923      0.987      0.636

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   137/149     1.71G   0.02223   0.01278   0.01611        29       416: 100% 4/4 [00:00<00:00,  6.66it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.70it/s]
                 all         18         18      0.867      0.971      0.979      0.606

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   138/149     1.71G   0.01687   0.01243   0.01784        26       416: 100% 4/4 [00:00<00:00,  6.74it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.52it/s]
                 all         18         18      0.977      0.932      0.987      0.617

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   139/149     1.71G    0.0239   0.01319   0.01698        32       416: 100% 4/4 [00:00<00:00,  6.85it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.05it/s]
                 all         18         18      0.978      0.933      0.987      0.622

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   140/149     1.71G   0.01946   0.01319   0.01707        24       416: 100% 4/4 [00:00<00:00,  6.13it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.57it/s]
                 all         18         18      0.971      0.937      0.987      0.625

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   141/149     1.71G    0.0199   0.01228    0.0151        26       416: 100% 4/4 [00:00<00:00,  6.67it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.06it/s]
                 all         18         18      0.968      0.944      0.987      0.621

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   142/149     1.71G   0.01784   0.01073   0.01375        24       416: 100% 4/4 [00:00<00:00,  6.37it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.99it/s]
                 all         18         18      0.953      0.944      0.987      0.636

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   143/149     1.71G   0.01925   0.01379    0.0134        29       416: 100% 4/4 [00:00<00:00,  6.49it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.72it/s]
                 all         18         18      0.874      0.965      0.979      0.595

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   144/149     1.71G   0.01959   0.01274   0.01802        24       416: 100% 4/4 [00:00<00:00,  6.41it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.99it/s]
                 all         18         18       0.88      0.963      0.979      0.614

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   145/149     1.71G    0.0279   0.01481   0.01819        35       416: 100% 4/4 [00:00<00:00,  5.95it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.94it/s]
                 all         18         18      0.886      0.963      0.979      0.621

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   146/149     1.71G   0.01795   0.01238    0.0159        31       416: 100% 4/4 [00:00<00:00,  6.38it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  9.38it/s]
                 all         18         18       0.89      0.963      0.979      0.625

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   147/149     1.71G   0.02616   0.01249   0.01851        35       416: 100% 4/4 [00:00<00:00,  6.25it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.77it/s]
                 all         18         18      0.889      0.964      0.979      0.625

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   148/149     1.71G   0.02275   0.01198   0.01715        36       416: 100% 4/4 [00:00<00:00,  6.79it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.66it/s]
                 all         18         18      0.891      0.964      0.979      0.624

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   149/149     1.71G   0.01744   0.01175   0.01528        30       416: 100% 4/4 [00:00<00:00,  6.45it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.93it/s]
                 all         18         18      0.891      0.965      0.979      0.632

150 epochs completed in 0.049 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 14.3MB
Optimizer stripped from runs/train/exp/weights/best.pt, 14.3MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model summary: 213 layers, 7018216 parameters, 0 gradients, 15.8 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  6.38it/s]
                 all         18         18      0.931      0.932      0.987      0.666
             DESKTOP         18          6      0.958          1      0.995      0.556
            NOTEBOOK         18          6      0.835      0.842      0.972       0.51
          SMARTPHONE         18          6          1      0.954      0.995      0.933
Results saved to runs/train/exp
  ```
</details>

### Evidências do treinamento
**![image](https://user-images.githubusercontent.com/90226923/184630157-e4300519-75b6-4f34-b418-39f3ec937281.png)
  
  **
  
  ![image](https://user-images.githubusercontent.com/90226923/184630217-89f0fc77-55b9-4204-8f59-d9ee2360a55e.png)


## Roboflow
https://app.roboflow.com/ia-iqcs5/projeto-final---modelos-preditivos-conexionistas/browse
## HuggingFace
https://huggingface.co/robertosa13/images_modelos_conexionistas/tree/main
  
