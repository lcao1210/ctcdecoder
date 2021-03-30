# CTC解码器
[![Python Version](https://img.shields.io/badge/Python-3.6.9-blue.svg)](https://www.python.org/)

## 执行流程
利用setup.sh脚本安装swig_decoders包
```
cd decoder
chmod u+x setup.sh
./setup.sh
```

## 功能介绍
用于CTC解码，支持贪婪解码(greedy decode)以及束搜索解码(beam search decode)，使用方法详见`demo.py`

## 感谢
https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech/commit/d0ebd704b5b2599e269d1f9852a489ef928e468d
