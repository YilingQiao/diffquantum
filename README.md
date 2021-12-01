
# Differentiable Analog Quantum Computing for Learning and Control
This project can run on python3.8
## Install
```bash
git clone git@github.com:YilingQiao/diffquantum.git
cd diffquantum
pip install -r requirements.txt
```
## Demos
1. Run GRAPE using gradients computed by torch.
```bash
python grape.py
```
2. Compare my pulse-based simulation with qutip.
```bash
python  compare_our_grape_with_qutip.py
```
3. Use our method to optimize Fourier coefficient.
```bash
python  ours_spectral.py
```
