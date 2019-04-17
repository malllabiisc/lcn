## Lovász Convolutional Networks

Source code for [AISTATS 2019](https://www.aistats.org/) paper: [Lovász Convolutional Networks](https://arxiv.org/abs/1805.11365).

<img align="right"  src="./overview.png">

### Dependencies

- Compatible with TensorFlow 1.x and Python 3.x.
- Dependencies can be installed using `requirements.txt`.

### Dataset:

- The current code allows evaluation on synthetic datasets which can be downloaded from [here](https://drive.google.com/open?id=1dZY9cx6poEjzyPAlI194TklzwrC3-ORr).

### Evaluate pretrained model:

- Run `setup.sh` for setting up the environment and extracting the datasets and pre-trained models.
- `lcn.py` contains TensorFlow (1.x) based implementation of **LCN** (proposed method).
- Execute `evaluate.sh` for evaluating pre-trained **LCN** model on all four datasets.

### Training from scratch:

- Execute `setup.sh` for setting up the environment and extracting datasets. 

- For training **LCN** run:

  ```shell
  python lcn.py -data citeseer -name new_run -kernel <lovasz/kls/none>
  ```

### Citation
Please cite us if you use this code.

```tex
@article{DBLP:journals/corr/abs-1805-11365,
  author    = {Prateek Yadav and
               Madhav Nimishakavi and
               Naganand Yadati and
	       Shikhar Vashishth and
               Arun Rajkumar and
               Partha Talukdar},
  title     = {Lovasz Convolutional Networks},
  journal   = {CoRR},
  volume    = {abs/1805.11365},
  year      = {2018},
  url       = {http://arxiv.org/abs/1805.11365},
  archivePrefix = {arXiv},
  eprint    = {1805.11365},
  timestamp = {Mon, 13 Aug 2018 16:46:04 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1805-11365},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

For any clarification, comments, or suggestions please create an issue or contact [shikhar@iisc.ac.in](http://shikhar-vashishth.github.io).
