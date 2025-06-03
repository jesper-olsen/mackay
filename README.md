## MacKay

Reimplemntation of some of the demos David MacKay gives in his Information Theory course [1][2].
The original demos are in Tcl/octave/gnuplot/perl/python - some of them survive on his website [3].
The Tcl demos generally still work as do the few perl & python demos (non graphical).
The octave/gnuplot demos, however, do not work with modern versions of octave and gnuplot.
Also - the demos exist in different versions tailord for different environments, e.g. labeled
Lewis and Skye, which probably were different servers in the engineering department.

References
----------

[1] [Video Lectures - MacKay's Information Theory course](https://videolectures.net/videos/)

[2] [Information Theory, Inference, and Learning Algorithms, David J.C. MacKay](https://www.inference.org.uk/mackay/Book.html)

[3] [MacKay's original perl/gnuplot implementation](https://www.inference.org.uk/mackay/itprnn/code/)

DEMOS
-----

The demos are in python3 and make use of numpy, scipy and matplotlib, e.g. install locally with the [uv](https://github.com/astral-sh/uv) package manager:

```
% uv venv
% uv pip install matplotlib numpy
```

* Chapter 21/Lecture 9: [Infer Gaussian](InferG.md)
* Chapter 29/Lecture 13: [Metropolis Simulation (Bonk!)](READMEbonk.md)
* Chapter 29/Lecture 12: [MC Importance Sampling]()
* Chapter 29/Lecture 12: [MC Rejection Sampling]()
* Chapter 29/Lecture 12: [MCMC Metropolis Sampling]()
* Chapter 29/Lecture 12: [MCMC Slice Sampling]()
* Chapter29/Lecture 12: [MCMC Gibbs Sampling]()
