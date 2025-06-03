Metropolis Simulation (Bonk)
=====================

Metropolis simulation of a system with `L = 21` discrete states.

- **Target distribution**: uniform  
- **Proposal distribution**: symmetric (`p = 0.5` left, `0.5` right)  
- **Boundary condition**: rejection at edges  
- **Step size**: `ε = 1`  
- **Total time steps**: `T`

### Variance Growth:
```
⟨Δx²⟩ ∝ T
```

### Expected Time to Hit a Wall:
```
 ⟨x²⟩ ≈ L² => T ≈ (L/ε)²
```

References
----------

[1] [Lecture 13 from David MacKay's Information Theory course](https://videolectures.net/videos/mackay_course_13)

[2] Chapter 29, [Information Theory, Inference, and Learning Algorithms, David J.C. MacKay](https://www.inference.org.uk/mackay/Book.html)

[3] [MacKay's original perl/gnuplot implementation](https://www.inference.org.uk/mackay/itprnn/code/metrop/)


Run
---

Random walk in a corridor of width 20 - draw histogram of positions visited.
```
% uv venv
% uv run metropolis_demo.py
Pausing every 20 iterations - press enter to continue
|           *         |
|            *        |
|             *       |
|            *        |
|             *       |
|              *      |
|               *     |
|              *      |
|               *     |
|                *    |
|                 *   |
|                *    |
|                 *   |
|                  *  |
|                   * |
|                    *| Bonk!
|                    *| Bonk!
|                    *| Bonk!
|                   * |
|                  *  |
------- at t=20 -------
|                   * |
|                  *  |
|                 *   |
|                *    |
|                 *   |
|                  *  |
|                   * |
|                    *| Bonk!
|                    *| Bonk!
|                   * |
|                  *  |
|                 *   |
|                  *  |
|                 *   |
|                  *  |
|                 *   |
|                  *  |
|                 *   |
|                  *  |
|                   * |
------- at t=40 -------
```
![PNG](https://raw.githubusercontent.com/jesper-olsen/mackay/main/Assets/Figure_1.png)

Position 10 and position 17 occupied initially.
Display probability of being in each position at time step t.
```
% uv run metropolis_demo_prob.py
Press Enter for next step...
Press Enter for next step...
Press Enter for next step...
Press Enter for next step...
```
![PNG](https://raw.githubusercontent.com/jesper-olsen/mackay/main/Assets/Figure_2.png)
