# SHA-KG

This repository contains code for the paper "Deep Reinforcement Learning with Stacked Hierarchical Attention for Text-based Games" by Yunqiu Xu, Meng Fang, Ling Chen, Yali Du, Joey Tianyi Zhou and Chengqi Zhang

## Dependencies

+ Python 3.7
    + pytorch 1.3.1
    + gym 0.17.2
    + jericho 2.4.0
    + networkx 2.4
    + redis 3.4.1
+ Redis 4.0.14 
+ Standford CoreNLP 3.9.2


## How to train

+ Modify the port number for redis (default 6381) and corenlp (default 9010) in ``env.py``, ``openie.py`` and ``vec_env.py``
+ Modify the path of corenlp in ``train_shakg.py``
+ Launch redis and corenlp
+ Run the code
```python

python train_shakg.py
```

## Citation

Coming soon


## Acknowledgement

We thank [rajammanabrolu/KG-A2C](https://github.com/rajammanabrolu/KG-A2C) for providing the excellent codebase. 

## License

[MIT License](https://github.com/YunqiuXu/SHA-KG/blob/main/LICENSE)
