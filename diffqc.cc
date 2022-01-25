#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <stdio.h>
#include <vector>  
// #include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <math.h>
#include <cmath>


// typedef float Scalar;
// typedef Eigen::MatrixXcf Matrixc;
// typedef Eigen::VectorXcf Vectorc;
typedef double Scalar;
typedef Eigen::MatrixXcd Matrixc;
typedef Eigen::VectorXcd Vectorc;
typedef std::complex<Scalar> Complex;

Matrixc g_H0;
std::vector<Matrixc> g_Hs;
std::vector<std::vector<std::vector<Scalar>>> g_channels;
Scalar g_duration;

void print_test () {
    std::cout << "hello\n";
}

std::vector<std::complex<Scalar>> 
complex_test(std::vector<std::complex<Scalar>> psi0) {
    return psi0;
}

std::vector<std::vector<double>>  test_eigen(std::vector<std::vector<double>> v) {
    return v;
}

// trotter(H_, psi0_, T0, T, n_steps=None)
// trotter(H0, Hs, psi0, T0, T, n_steps=None)

void set_H(
    std::vector<std::vector<Complex>> _H0,
    std::vector<std::vector<std::vector<Complex>>> _Hs,
    std::vector<std::vector<std::vector<Scalar>>> channels,
    Scalar duration) {
    int n_qubit = _H0.size();

    Matrixc H0(n_qubit, n_qubit);

    for (int i = 0; i < n_qubit; ++i)
        for (int j = 0; j < n_qubit; ++j)
            H0(i, j) = _H0[i][j];

    g_H0 = H0;

    g_Hs.clear();
    for (int k = 0; k < _Hs.size(); ++k)
    {
        Matrixc Hs(n_qubit, n_qubit);
        for (int i = 0; i < n_qubit; ++i)
            for (int j = 0; j < n_qubit; ++j)
                Hs(i, j) = _Hs[k][i][j];
        g_Hs.push_back(Hs);
    }

    g_channels = channels;
    g_duration = duration;
}

Scalar my_expit(Scalar x) {
    Scalar cutoff = 32.;
    if (x > cutoff) return 1.;
    if (x < -cutoff) return 0.;
    return 1 / (1 + std::exp(-x));
}

Scalar f_u(
    int h, Scalar t,
    std::vector<std::vector<std::vector<Scalar>>>& vv
    ) {

    Scalar ans = 0;
    int n_basis = vv[0][0].size();

    for (int i_c = 0; i_c < g_channels[h].size(); i_c++) {
        Scalar A = 0;
        Scalar B = 0;
        Scalar N = 0;

        std::vector<Scalar>& chan = g_channels[h][i_c];
        Scalar omega = chan[1];
        Scalar w = chan[2];
        int idx = round(chan[3]); 

        for (unsigned int j = 0; j < n_basis; ++j)
        {
            A += vv[0][idx][j] * std::legendre(j, 2 * t / g_duration - 1);
            B += vv[1][idx][j] * std::legendre(j, 2 * t / g_duration - 1);
        }
        N = std::sqrt(A * A + B * B);
        if (abs(N-0.0) < 0.000001) {
            ans += 0.0;
        } else {
            ans += omega * (2 * my_expit(N) - 1) / N * (cos(w * t) * A + sin(w * t) * B);
        }
    }
    return ans;
}

/*
std::vector<Complex> trotter(
    std::vector<Complex>& _psi0,
    Scalar T0,
    Scalar T,
    int per_step,
    std::vector<std::vector<std::vector<Scalar>>>& vv
    ) {
    
    int n_qubit = _psi0.size();
    int n_steps = (int) (per_step * (std::abs(T - T0) + 1));
    Scalar dt = (T - T0) / n_steps;
    Scalar t = T0;

    Complex i_unit = 1.i;

    Vectorc psi0 = Eigen::Map<Vectorc, Eigen::Unaligned>(_psi0.data(), _psi0.size());
    
    for (int step = 0; step < n_steps; ++step)
    {
        psi0 = (- i_unit * dt * g_H0).exp() * psi0;
        for (int h = 0; h < g_Hs.size(); ++h)
        {
            Scalar coeff = f_u(h, t, vv);
            psi0 = (- i_unit * dt * coeff * g_Hs[h]).exp() * psi0;
        }
        t += dt;
    }

    std::vector<Complex> out_psi0(psi0.data(), psi0.data() + psi0.size());

    return out_psi0;
}
*/

///*
std::vector<Complex> trotter(
    std::vector<Complex>& _psi0,
    Scalar T0,
    Scalar T,
    int per_step,
    std::vector<std::vector<std::vector<Scalar>>>& vv
    ) {
    
    int n_qubit = _psi0.size();
    int n_steps = (int) (per_step * (std::abs(T - T0) + 1));
    Scalar dt = (T - T0) / n_steps;
    Scalar t = T0;

    Complex i_unit = 1.i;

    Vectorc psi0 = Eigen::Map<Vectorc, Eigen::Unaligned>(_psi0.data(), _psi0.size());
    
    for (int step = 0; step < n_steps; ++step)
    {
			Matrixc dH = - i_unit * dt * g_H0;
			for (int h = 0; h < g_Hs.size(); ++h)
        {
					Scalar coeff = f_u(h, t, vv);
					dH += - i_unit * dt * coeff * g_Hs[h];
				}
			psi0 = dH.exp() * psi0; 
			t += dt;
    }

    std::vector<Complex> out_psi0(psi0.data(), psi0.data() + psi0.size());

    return out_psi0;
}
//*/

namespace py = pybind11;

PYBIND11_MODULE(diffqc, m) {
    m.doc() = R"pbdoc(
        differentiable quantum computing
        -----------------------

        .. currentmodule:: diffqc

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    m.def("print_test", &print_test);
    m.def("complex_test", &complex_test);
    m.def("test_eigen", &test_eigen);
    m.def("trotter", &trotter);
    m.def("set_H", &set_H);
    m.attr("__version__") = "dev";
}
