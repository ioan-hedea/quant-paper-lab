# Julia Project Setup

This folder contains Julia scripts for computing polynomial certificates using
sum-of-squares (SOS) optimization.

The code relies on several Julia packages and should be run inside the provided
project environment so that all dependencies are installed automatically.

---

# 1. Installing Julia

If you do not already have Julia installed:

1. Download Julia from: [https://julialang.org/downloads/](https://julialang.org/downloads/)
2. Install it using the default settings for your operating system.

After installation, verify that Julia works by running in a terminal:

```
julia
```

This should open the Julia REPL.

---

# 2. Activating the project environment

Open a terminal in the folder containing this project and start Julia:

```
julia
```

Then activate the local project and install the required packages:

```
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

`Pkg.instantiate()` installs all dependencies specified in the project
configuration files.

The main packages used in this project include:

* JuMP
* SumOfSquares
* DynamicPolynomials
* MultivariatePolynomials
* SCS

The first installation may take a few minutes.

---

# 3. Running the scripts

After completing the TODO blocks in the code, the scripts can be run from the Julia REPL using `include`.

Example:

```
include("lyapunov_certificate.jl")
```

or 

```
include("barrier_certificate.jl")
```


Each script synthesizes a polynomial certificate and then visualizes the result.

