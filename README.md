
# Differentiable Analog Quantum Computing for Learning and Control
This project can run on python3.8

## Install
```bash
git clone git@github.com:YilingQiao/diffquantum.git
cd diffquantum
pip install -r requirements.txt
```

### Python binding
Build the python binding 
```bash
git submodule init
git submodule update
python setup.py install
```

## Demos
1. Run our implementation of IBMQ simulator. 
```bash
python main_ibmsim.py
```
2. Run simulation without modulation.
```bash
python main_nomod.py
```
3. Run GRAPE. 
```bash
python main_grape.py
```