# Neural-network controllers

## Installation Instructions

1. Install version `1.8.5` of Julia:  
   
   A. If you don't have Julia on your system yet, you can directly download the [old release](https://julialang.org/downloads/oldreleases/).   
   B. If `1.8.5` is the only version on your system, you can open the Julia REPL by simply running `julia` in terminal.  
   C. Otherwise, you can use the `juliaup` utility in terminal by running `juliaup add 1.8.5` and then `julia +1.8.5` to launch REPL with a specific version.

2. Once you have Julia REPL open:
 
    A. Move to the `assignment2` directory.
    B. Press `]` to open `Pkg`, the Julia package manager.  
    C. Type `activate .` and press `Enter` to create a new environment.  
    D. Press `backspace` to return to the Julia REPL.  
    E. Type in `include("requirements.jl")` and press `Enter`.  
    F. Note that installing the dependencies and precompiling the project make take 15-30 minutes.

**Note:** you can also directly instantiate the package with `using Pkg; Pkg.instantiate()` in Julia REPL.

3. Once all packages have been installed, you can run the assignment code with the following commands in the Julia REPL:

    A. ```include("Single-Pendulum.jl")```  
    B. ```include("Double-Pendulum.jl")```

If you find any bugs, please let us know. This way, you can help us improve the assignment for next year's students.