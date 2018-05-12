# fast-image-style-transfy

说明：已将环境迁移到了python3.6，TensorFlow1.8

## macos 环境部署命令
```
    brew install python3

    sudo easy_install pip

    pip install tensorflow

    cd fast-image-style-transform/

    python3 web.py
```

## 如何训练自己的模型：

首先下载vgg_16模型, 官网链接(http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz). 在祝目录下建立pretrained并解压得到vgg_16.ckpt。

然后下载训练数据（里面是大量的高清晰度的摄影图片资料)链接: https://pan.baidu.com/s/1baokxDMm_F4iNz6iD6OhLQ 密码: r672
. 解压到文件夹内。

示例：
训练"wave"模型:
```
    python train.py -c conf/wave.yml
```
  
##### (Optional) Use tensorboard:
```
    tensorboard --logdir models/wave/
```
    
具体参数设置见 "conf/wave(https://github.com/ousc/fast-image-style-transform/conf/wave.yml)".
