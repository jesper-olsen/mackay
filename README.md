Mackay
==============

This is a python implementation of the metropolis demo David Mackay gives in the first part of [1].

References
----------

[1] [Lecture 12 from David MacKay's Information Theory course](https://videolectures.net/videos/mackay_course_12)

[2] Chapter 29, [Information Theory, Inference, and Learning Algorithms, David J.C. MacKay](https://www.inference.org.uk/mackay/Book.html)



Run
---

Random walk in a corridor of width 20 - draw histogram of positions visited.
```
% uv venv
% uv run metropolis_demo.py
% uv pip install matplotlib numpy
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
