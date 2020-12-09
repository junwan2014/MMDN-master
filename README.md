# MMDN-master
A pytorch implementation of "Robust Facial Landmark Detection by Multi-order Multi-constrained Network"
# Training 
1. change the path of training and testing datasets to your own paths in main.py. 
```
-imgdirs_train = ['D:/dataset/300W/300W_LP/300W/']
-imgdirs_test_commomset = ['D:/dataset/300W/300W_LP/ibug/']
```

2. python ./main.py

# Testing
1. change the path of training and testing datasets to your own paths in demo.py. 
```
imgdirs_test_commomset = ['D:/dataset/ibug/']
 ```

2. python ./demo.py
3. If you want to use the funtion get_subpixel_from_kpts() to accelerate the testing in demo.py, then you should
```
cd ./MMDN-master
python setup.py build_ext --inplace
```
# Reference
1. If the the work or the code is helpful, please cite the following papers
```
Jun Wan, Zhihui Lai, Jing Li, Jie Zhou, Can Gao, “Robust Facial Landmark Detection by
Multi-order Multi-constraint Deep Networks", IEEE Transactions on Neural Networks and
Learning Systems.
```
```
Jun Wan, Zhihui Lai, Jun Liu, Jie Zhou, Can Gao, “Robust Face Alignment by Multi-order
High-precision Hourglass Networks", IEEE Transactions on Image Processing, 2021, 30, pp.
121-133.
```
```
Jun Wan, Zhihui Lai, Jie Zhou, Can Gao, Jing Li, “Robust Facial Landmark Detection by
Cross-order Cross-semantic Deep Network", Neural Networks, https://doi.org/10.1016/j.neun-
et.2020.11.001.
```

