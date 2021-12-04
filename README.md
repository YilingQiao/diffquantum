
# Differentiable Analog Quantum Computing for Learning and Control
This project can run on python3.8

## TODO
- [x] GRAPE energy minimization and fidelity 
- [x] Ours (Fourier) energy minimization - Control
- [ ] Ours (Fourier) fidelity - Control
- [ ] Parameter Shift
- [ ] Ours (Parameter Shift) - Learning
- [ ] Ours numerical integration

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
2. Compare the forward simulation results of my pulse-based simulation with that of qutip.
```bash
python  compare_our_grape_with_qutip.py
```
3. Use our method to optimize Fourier coefficient. You can compare the energy with the results in 1, which should be the same.
```bash
python  ours_spectral.py
```
