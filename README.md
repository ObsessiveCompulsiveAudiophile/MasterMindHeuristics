# MasterMindHeuristics
# GPU-Accelerated Mastermind Strategy Optimizer
<meta name="google-site-verification" content="KqNySZDgV7TztABTaUQROABWYp0Mep-Xv2oqE1NneR0" />

This repository contains a high-performance C++/CUDA application that uses a Genetic Algorithm (GA) to find the optimal coefficients for a Mastermind-playing strategy based on Shannon Entropy. The goal is to discover a set of weights that minimizes the average number of guesses required to solve any of the 1,296 possible Mastermind codes (6 colours of 4 peg combinations).

The entire population's fitness evaluation is parallelized on the GPU, allowing for tens of thousands of generations to be processed in a reasonable timeframe.

Screen output:
  
--- GPU-Accelerated Mastermind Optimizer ---
Setting up...
Pre-calculating log2 table...
Allocating memory on GPU...
Memory allocated.
Population Size: 4096, Generations: 50000
Starting search... Press Ctrl+C to stop early.

T+86.29s Gen 1: !!! NEW BEST: 4.45216049 !!!
  Coefficients: [ 2.039864 1.793214 0.755637 1.785964 1.906162 0.679069 0.773728 0.806738 1.441272 1.604575 1.866051 1.709619 1.777104 2.497700 ]
  
T+210.89s Gen 2: !!! NEW BEST: 4.37268519 !!!

  Coefficients: [ 1.011635 0.963732 1.134579 0.750452 0.998251 1.107951 1.152932 1.087969 0.843971 1.049103 1.084014 0.845030 0.925238 2.052436 ]
  
T+334.61s Gen 3: !!! NEW BEST: 4.36265432 !!!

  Coefficients: [ 1.217594 1.091760 1.140076 0.806293 1.005788 1.318068 1.291859 1.276621 0.990647 1.171217 1.085449 0.954608 1.192238 2.278548 ]
  
T+701.16s Gen 6: !!! NEW BEST: 4.36188272 !!!

  Coefficients: [ 1.134729 1.073169 1.066672 0.787112 0.979749 1.312589 1.253307 1.120857 0.945278 1.068803 1.077165 0.890370 0.937879 2.328730 ]
  
T+946.07s Gen 8: !!! NEW BEST: 4.36111111 !!!

  Coefficients: [ 1.141190 1.111299 1.140128 0.819053 0.962552 1.359213 1.365484 1.222252 0.944071 1.103748 1.085225 0.905094 1.076151 2.234329 ]
  
T+1068.78s Gen 9: !!! NEW BEST: 4.36033951 !!!

  Coefficients: [ 1.192595 1.241326 1.197373 0.710006 0.982864 1.162020 1.084940 1.073871 0.908764 0.993957 0.853172 0.961896 1.220856 2.291452 ]
  
T+1437.68s Gen 12: !!! NEW BEST: 4.35956790 !!!

  Coefficients: [ 1.130136 1.223617 1.238801 0.979487 0.920871 1.409870 1.164528 1.077106 1.018219 1.020494 1.066287 1.261417 1.010666 2.387494 ]
  
T+2423.80s Gen 20: !!! NEW BEST: 4.35802469 !!!

  Coefficients: [ 1.253748 1.116675 1.244436 0.875827 0.908599 1.355589 1.085168 0.984490 1.002644 1.039383 0.987588 1.155333 0.993035 2.203433 ]
  
Gen 500: Current best: 4.35802469 (Best ever: 4.35802469) Stagnation: 480

T+108951.25s Gen 875: !!! NEW BEST: 4.35725309 !!!

  Coefficients: [ 1.291028 1.415892 1.352769 0.886042 0.957370 1.253160 1.296962 1.202261 1.056080 1.152413 1.164210 1.357928 0.970690 2.469986 ]
  
...


Note: A full game tree average of 4.3565 has been achieved (in less then 48 hours of training) and this is a heuristic strategy "World record" for MasterMind game and ranks only second after Koyoma & Lai's brute force DFS optimal algorithm (4.3403)

## The Strategy

The core of the puzzle-solving strategy is to select the "best" guess at each turn. "Best" is defined as the guess that, on average, provides the most information, thereby reducing the set of possible solutions by the greatest amount. This is calculated using a weighted form of Shannon's entropy formula:

`Score(Guess) = Σ ( P(mark) * -log₂(P(mark)) * Coef(mark) )`

Where:
-   **`Guess`**: One of the 1,296 possible codes.
-   **`mark`**: A possible feedback response (e.g., "1 black peg, 2 white pegs"). There are 14 possible marks.
-   **`P(mark)`**: The probability of receiving that mark for the given `Guess` against all remaining possible solutions.
-   **`Coef(mark)`**: A coefficient (or weight) for that specific mark type.

The goal of this project is to **find the optimal set of 14 coefficients** that leads to the best overall average score.

## The Genetic Algorithm

A Genetic Algorithm is used to explore the vast "solution space" of possible coefficient sets. The process is as follows:

1.  **Initialization**: A large population (e.g., 4096) of "individuals" is created. Each individual consists of a set of 14 floating-point coefficients.
2.  **Fitness Evaluation (The CUDA Kernel)**: The fitness of every individual in the population is calculated. This involves simulating every possible game (1,296 of them) for that individual's set of coefficients and calculating the average number of guesses needed. This computationally massive step is offloaded entirely to the GPU, where each individual in the population is evaluated in parallel.
3.  **Selection**: The fittest individuals (those with the lowest average score) are selected as "parents" for the next generation. A portion of the very best parents (**Elitism**) are preserved and copied directly to the next generation to ensure the best-found solutions are never lost.
4.  **Reproduction & Mutation**: The next generation is created by performing "crossover" (combining the coefficients of two parents) and "mutation" (making small, random changes to the coefficients). The strength of the mutation adaptively increases if the algorithm's progress stagnates, helping it to break out of local optima.
5.  **Repeat**: The process repeats for a large number of generations (e.g., 50,000+), continuously evolving the population toward better solutions.

## Performance & Optimization

This project employs several advanced optimization techniques to achieve high performance:
-   **Massive Parallelism**: The `evaluatePopulationKernel` runs the most expensive part of the GA (fitness evaluation) on thousands of GPU cores simultaneously.
-   **`log2f` Lookup Table**: Instead of calculating `log2f` repeatedly, values are pre-calculated on the host and stored in a GPU-side lookup table.
-   **Read-Only Cache (`__ldg`)**: The lookup table is read using the `__ldg()` intrinsic, which pulls data through the GPU's dedicated read-only cache. This prevents cache contention with other data, providing a noticeable speedup over naive global memory reads.
-   **Adaptive Mutation**: The mutation strength is dynamically adjusted based on performance, preventing premature convergence and encouraging a more thorough search of the solution space.

## Requirements

-   A modern NVIDIA GPU (Kepler architecture or newer for `__ldg()` support).
-   [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (tested with version 11.x/12.x).
-   A C++ compiler (e.g., MSVC on Windows).
-   Visual Studio (recommended on Windows for easy project setup).

## Building and Running

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
    cd YOUR_REPOSITORY
    ```

2.  **Open and Build:**
    -   **Windows / Visual Studio**: Open the `.sln` solution file. Set the build configuration to `Release` and the platform to `x64`. Build the solution (F7 or Build -> Build Solution).
    -   **Linux**: You can compile from the command line using `nvcc`:
        ```sh
        nvcc -O3 -arch=sm_XX --std=c++17 -o optimizer cudaKernel.cpp
        ```
        *(Note: Replace `sm_XX` with the compute capability of your GPU, e.g., `sm_75` for a Turing card or `sm_86` for an Ampere card)*

3.  **Run the executable:**
    -   From the Visual Studio build output directory (`x64/Release`) or the directory where you compiled the binary, simply run the executable.
    ```sh
    ./optimizer.exe  # Windows
    ./optimizer      # Linux
    ```
    The program will start the optimization process and print updates to the console whenever a new best score is found.

## Results & Future Work

The optimizer can discover highly effective coefficient sets that achieve near-optimal average scores. The console log shows the progress over time, tracking the best score found and the number of generations without improvement ("Stagnation").

****Further applied the below improvements on Jul 12 2025****
-   **Adaptive Mutation Parameters**
-   **Enhanced Crossover and Mutation Kernel**
-   **Population Diversity Tracking**
-   **Improved Initialisation**
  
NEW OUTPUT:

Allocating memory on GPU...
Memory allocated.
Population Size: 4096, Generations: 50000
Starting search...

T+114.84s Gen 1: !!! NEW BEST: 4.35648148 !!!
  Coefficients: [ 1.24 1.34 1.35 0.95 0.86 1.25 1.33 1.17 1.03 1.10 1.13 1.30 1.05 2.32 ]
  
  [Diversity: 51.72, Stagnation: 150] Increasing mutation: rate=0.216, amount=16
  
  [Diversity: 66.24, Stagnation: 150] Increasing mutation: rate=0.300, amount=17
  
  T+40924.52s Gen 333: !!! NEW BEST: 4.35570988 !!!
  
  Coefficients: [ 1.24 1.34 1.42 1.03 0.79 1.25 1.33 1.17 1.00 1.08 1.11 1.33 1.10 2.27 ]
  
  

