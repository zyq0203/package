# Source code —— FDCNet: Frontend-backend fusion dilated network through channel-attention mechanism
This is an implementation of paper "FDCNet: Frontend-backend fusion dilated network through channel-attention mechanism"


1.Install pytorch

2.Clone this repository



# Data Setup
1. Download ShanghaiTech Dataset from
Baidu Disk: http://pan.baidu.com/s/1nuAYslz

2.Create Directory
mkdir ROOT/data/original/shanghaitech/  

3.Save "part_A_final" under ROOT/data/original/shanghaitech/

4.Save "part_B_final" under ROOT/data/original/shanghaitech/

5.cd ROOT/data_preparation/
run create_gt_test_set_shtech.m in matlab to create ground truth files for test data

6.cd ROOT/data_preparation/
run create_training_set_shtech.m in matlab to create training and validataion set along with ground truth files



# Training
Follow steps 1,2,3,4 and 6 from Data Setup

Run train.py



# Test
Follow steps 1,2,3,4 and 5 from Data Setup

Save the model files under ROOT/final_models

Run test.py
