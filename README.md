# CL-Traff
An implementation of CL-Traff under GCRN backbone.
#### 2024/05/29
Fix a load adj bug for PEMS0X, doesn't affect the results.
Only appear after the D7 version, already fix now.

#### 2024/05/15

* Update PEMSD7L and PEMSD7M datasets

* Run following commands to prepare data:

  ```bash
  python generate_training_data_his_D7.py --dataset PEMSD7L
  python generate_training_data_his_D7.py --dataset PEMSD7M
  ```

* Then train the model with following commands:

  ```bash
  cd model_MDGCRN
  python traintorch_MDGCRNAdjHiDD.py --gpu 1 --dataset PEMSD7L
  python traintorch_MDGCRNAdjHiDD.py --gpu 1 --dataset PEMSD7M
  ```

  



#### 2024/04/19
The newest results are as follows:
![画像1](https://github.com/deepkashiwa20/CL-Traff/assets/54202779/cd76656f-cd5d-4c5d-92a8-4cc48ed04a44)

  ```bash
  cd model_MDGCRN
  python traintorch_MDGCRNAdjHiDD.py --gpu=0 --dataset=PEMS08
  ```

#### 2024/02/05

* Update PEMS0X dataset

* Run following commands to prepare data:

  ```bash
  python generate_training_data_his_PEMS.py --dataset PEMS03
  python generate_training_data_his_PEMS.py --dataset PEMS04
  python generate_training_data_his_PEMS.py --dataset PEMS07
  python generate_training_data_his_PEMS.py --dataset PEMS08
  ```

* Then train the model with following commands:

  ```bash
  cd model_MDGCRN
  python traintorch_MDGCRNAdjHiDD.py --gpu 1 --dataset PEMS03
  python traintorch_MDGCRNAdjHiDD.py --gpu 1 --dataset PEMS04
  python traintorch_MDGCRNAdjHiDD.py --gpu 1 --dataset PEMS07
  python traintorch_MDGCRNAdjHiDD.py --gpu 1 --dataset PEMS08
  ```

  

#### Latest Release
* 生成PEMS-BAY的数据 (Finish)
* python generate_training_data_his_BAY.py
* 生成EXPYTKY的数据 (Finish)
* python generate_training_data_his_EXPYTKY.py
* 实现MDGCRNAdjHiDD.py
* cd model_MDGCRN & python traintorch_MDGCRNAdjHiDD.py --gpu 0
* 实现MDGCRN_EXPYTKY
* cd model_MDGCRN_EXPYTKY & python traintorch_MDGCRNAdjHiDD.py --gpu 0

#### Latest Release1
* cd model_MDGCRN
* python traintorch_MDGCRNAdjHiD.py --gpu=0 --lamb xxx --lamb2 xxx --schema xxx
* 如果不使用对比学习, 设置--lamb 0, 即只使用MAE loss + detection loss, 有3种detection loss实现方式:
* python traintorch_MDGCRNAdjHiD.py --gpu=0 --lamb2 1.0 --schema 1 (方式1:这个hypernet的2个输入可以相加/相减/拼接，然后根据值域范围确定一个激活函数sigmoid确保在[0, 1])
* python traintorch_MDGCRNAdjHiD.py --gpu=0 --lamb2 1.0 --schema 2 (方式2: 计算2个anchor的(1 - cosine(x1, x2) ) / 2，[0, 1])
* python traintorch_MDGCRNAdjHiD.py --gpu=0 --lamb2 1.0 --schema 3 (方式3: 计算2个anchor的(1 - cosine(mlp(x1), mlp(x2)) ) / 2，[0, 1])
* 如果使用对比学习，设置--lamb 0.1, 即使用MAE loss + detection loss + contra loss: 有2种contra loss以及3种schema, 总共6种实现方式:
* python traintorch_MDGCRNAdjHiD.py --gpu=0 --lamb 0.1 --lamb2 1.0 --contra_loss infonce --schema xxx
* python traintorch_MDGCRNAdjHiD.py --gpu=0 --lamb 0.1 --lamb2 1.0 --contra_loss triplet --schema xxx

#### Updates (2023/10/20)
* git pull
* run the updated METRLA/preprocess_mean_nonullval_timeinday.ipynb
* turn off the notebook, otherwise the h5 file will be occupied.
* run the updated generate_training_data_plus.py
* we can get trainplus.npz, valplus.npz, testplus.npz, where the second channel (ycov channel) is timeinday not weekdaytime. This will affect the performance of original DGCRN.
* run the original DGCRN model_DGCRN/trainplus_DGCRN.py with new generated *plus.npz
* retest some techniques on the new program and the new data.

#### Requirements
* Python 3.8.8 -> Anaconda Distribution
* pytorch 1.9.1 -> py3.8_cuda11.1_cudnn8.0.5_0
* pandas 1.2.4 
* numpy 1.20.1
* tensorboard 2.12.1 -> conda install tensorboard
* setuptools 59.5.0 -> pip install setuptools==59.5.0
* torch-summary 1.4.5 -> pip install torch-summary https://pypi.org/project/torch-summary/ (must necessary)
* jpholiday -> pip install jpholiday (not must, but if you want onehottime)

#### Preparation
* For PEMSBAY dataset, please first upzip ./PEMSBAY/pems-bay.zip to get ./PEMSBAY/pems-bay.h5 file.
* Two trainers, one is traintest_GCRN.py inherited from [GTS](https://github.com/chaoshangcs/GTS), another is traintest+_GCRN.py.
* But traintest_GCRN.py may have a small bug as reported [here](https://github.com/deepkashiwa20/MegaCRN/issues/1#issuecomment-1445274957).
* For traintest_GCRN.py, please first run: python generate_training_data.py --dataset=DATA

#### Running
* cd model
* python traintest_GCRN.py or traintest+_GCRN.py --dataset=DATA --gpu=GPU_DEVICE_ID 
* DATA = {METRLA, PEMSBAY}
