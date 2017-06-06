环境说明

    操作系统: ubuntu 16.04

    cpu: Intel i5-7200u

    gpu: NVIDIA GeForce 940mx

    语言: python 3.6

    库: tensorflow 1.0, numpy, cuda 5.1, cudnn 8.0, gcc 4.9.3


数据集

    介绍 http://www.cs.toronto.edu/~kriz/cifar.html

    下载地址 http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz


运行方法

    下载数据集并安装必要的库.

    假设数据集解压后放在路径/home/cifar10/dataset

    假设本程序放在路径/home/cifar10/code

    假设训练后的模型数据将存放在路径/home/cifar10/train


    训练模型, 在命令行窗口依次输入

        cd /home/cifar10/code

        python cifar10_train_resnet.py --dataset_dir /home/cifar10/dataset --train_dir /home/cifar10/train

    输入后, 程序开始进行训练模型. 可以使用tensorboard进行监控训练过程, 在另一个命令行窗口输入

        tensorboard --logdir /home/cifar10/train

    当程序结束后, 即完成训练(在上述的环境中需要约20小时)
    
    假设模型的测试结果将存放在路径/home/cifar10/test
    
    测试模型, 在命令行窗口中依次输入
        
        cd /home/cifar10/code

        python cifar10_test_resnet.py --dataset_dir /home/cifar10/dataset --test_dir /home/cifar10/test

    经过数分钟, 程序结束, 并输出测试准确

        precision :0.911 

