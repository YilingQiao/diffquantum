
# Differentiable Analog Quantum Computing for Learning and Control
This project can run on python3.8

## TODO
- [x] GRAPE energy minimization and fidelity 
- [x] Ours (Fourier) energy minimization - Control
- [x] Ours (Legendre) energy minimization - Control
- [x] Comparison with QAOA
- [ ] Fidelity loss
- [ ] Comparison with parameter shift - learning
- [ ] Ours numerical integration

## Install
```bash
git clone git@github.com:YilingQiao/diffquantum.git
cd diffquantum
pip install -r requirements.txt
```
## Demos
1. Compare wit GRAPE on energy minimization. 
```bash
python plot_grape_ours.py
```
2. Compare wit QAOA on max-cut. 
```bash
python plot_qaoa_ours.py
```
