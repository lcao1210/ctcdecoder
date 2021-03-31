# CTC解码器
[![Python Version](https://img.shields.io/badge/Python-3-blue.svg)](https://www.python.org/)

## 功能介绍
用于CTC解码，支持贪婪解码(greedy decode)以及束搜索解码(beam search decode)

## 执行流程
利用setup.sh脚本安装swig_decoders包，安装过程如下，安装完成后即可直接调用`./decoders/ctcsecoder/swig_wrapper.py`中的相关贪婪解码函数以及束搜索解码函数，调用方法参见`demo.py`
```
cd decoder
chmod u+x setup.sh
./setup.sh
```

## KenLM训练
对于束搜索解码，需要向解码器中添加KenLM语言模型，可以通过下载已有公开模型或自行训练的方式获得：
### 已有公开模型：
可以使用已经训练的公开KenLM模型，约2.8G
```
mkdir lm
cd lm
wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
```

对于有足够大性能的机器（内存大于70G），可以下载70G的超大语言模型
```
mkdir lm
cd lm
wget https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm
```
### 自行训练：
除了使用已有公开语言模型，还可以将自己收集到的语料进行训练
1. 首先下载KenLM包：
```
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j2
```
如果在cmake中boost部分报错，执行：
```
sudo apt-get install libboost-all-dev
```

2. 进入build/目录，执行操作：
```
bin/lmplz -o 3 --verbose header --text test.txt --arpa test.arpa
```
其中：
```
-o 为必须的参数，3表示3-gram
–verbose header表示向ARPA文件开头添加详细的头文件（可选）
–text 训练的语料库，每个句子一行，每个字或词用空格隔开，最好不要包含标点符号以及阿拉伯数字
–arpa 训练生成的语言模型，在这之后为了减少空间可以将其变成二进制格式
```

3. 为了节省磁盘存储空间，将 arpa 文件转换为trie二进制文件
```
bin/build_binary trie -a 22 -q 8 -b 8 test.arpa test.klm
```
其中：
```
-a 表示在trie中用于切分的指针的最高位数。
-q 表示概率
-b 表示退避的量化参数
```

## 感谢
https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech

参考版本：
https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech/tree/d0ebd704b5b2599e269d1f9852a489ef928e468d