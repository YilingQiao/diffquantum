
# Differentiable Analog Quantum Computing for Learning and Control

[Jiaqi Leng*](https://jiaqileng.github.io/), [Junbang Liang*](https://pickspeng.github.io/), [Yi-Ling Qiao*](https://ylqiao.net/), [Ming C. Lin](https://www.cs.umd.edu/~lin/), [Xiaodi Wu](https://www.cs.umd.edu/~xwu/)
 [[arXiv]](https://github.com/YilingQiao/diffquantum) [[GitHub]](https://github.com/YilingQiao/diffquantum)

## Setup
We have tested our code on Ubuntu and Mac (some code needs to be modified for Mac) with Python 3.8. Below is an example of how to build this library
```bash
git clone git@github.com:YilingQiao/diffquantum.git
cd diffquantum
pip install -r requirements.txt
git submodule init
git submodule update
python setup.py install
```

## Demos
1. Run QAOA to solve the maxcut problems. 
```bash
python demo_maxcut.py
```
TODO. We are clearning the code for
2. Run VQE to solve for H2 Ground State and Energy.
3. Demos for quantum control problems.
4. Commparisons and draw

## Documentation 
TODO. We are preparing for more detailed docs.

## Bibtex
```
@inproceedings{leng2022diffaqc,
  title={Differentiable Analog Quantum Computing for Optimization and Control},
  author={Leng, Jiaqi and Peng, Yuxiang and Qiao, Yi-Ling and Lin, Ming and Wu, Xiaodi},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```