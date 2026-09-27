# UCB CS285: Deep Reinforcement Learning

Slides, sections and homework follow Spring 2026.

- Spring 2026: https://rail.eecs.berkeley.edu/deeprlcourse/

## Layout

| Path                            | Content                                            | In git |
| ------------------------------- | -------------------------------------------------- | ------ |
| `Slides/`                       | Lecture slides `lec-1` – `lec-25`                  | No     |
| `Sections/`                     | Discussion section slides                          | No     |
| `Homework/Handouts/`            | Homework handouts `hw1` – `hw5`                    | Yes    |
| `Homework/homework_spring2026/` | Official starter code, see below                   | Yes    |
| `Project/`                      | Final project outline and default project handouts | Yes    |

- `homework_spring2026/` is [berkeleydeeprlcourse/homework_spring2026][hw-repo] at `59e40fc`, under its MIT License. Its `final_project_llm_rl/dataset/` is gitignored.

Slide links point to the official site. Local copies use the same file names.

## Lectures

| #  | Topic                            | Slides        |
| -- | -------------------------------- | ------------- |
| 1  | Introduction                     | [slides][l01] |
| 2  | Behavioral Cloning               | [slides][l02] |
| 3  | Behavioral Cloning Part 2        | [slides][l03] |
| 4  | RL Basics                        | [slides][l04] |
| 5  | Policy Gradients                 | [slides][l05] |
| 6  | Actor Critic                     | [slides][l06] |
| 7  | Value-Based RL                   | [slides][l07] |
| 8  | Q-learning in Practice           | [slides][l08] |
| 9  | Advanced Policy Gradients Part 1 | [slides][l09] |
| 10 | Advanced Policy Gradients Part 2 | [slides][l10] |
| 11 | Variational Inference            | [slides][l11] |
| 12 | VI in RL                         | [slides][l12] |
| 13 | Control as Inference             | [slides][l13] |
| 14 | LLM RL                           | [slides][l14] |
| 15 | Model-Based RL Part 1            | [slides][l15] |
| 16 | Model-Based RL Part 2            | [slides][l16] |
| 17 | Offline RL Part 1                | [slides][l17] |
| 18 | Offline RL Part 2                | [slides][l18] |
| 19 | Exploration                      | [slides][l19] |
| 20 | RL Theory                        | [slides][l20] |
| 21 | Midterm Review Part 1            | [slides][l21] |
| 22 | Midterm Review Part 2            | [slides][l22] |
| 23 | Advanced Exploration             | [slides][l23] |
| 24 | Multi-task RL                    | [slides][l24] |
| 25 | Challenges and Open Problems     | [slides][l25] |

## Sections

| #   | Topic                             | Slides           |
| --- | --------------------------------- | ---------------- |
| 1   | PyTorch Tutorial                  | [slides][sec1]   |
| 2-1 | Probability Review                | [slides][sec2-1] |
| 2-2 | BC Distributional Shift           | [slides][sec2-2] |
| 3   | Policy Gradients and Actor Critic | [slides][sec3]   |
| 4   | DQN and SAC                       | [slides][sec4]   |
| 5   | Advanced Policy Gradients         | [slides][sec5]   |
| 6   | Variational Inference             | [slides][sec6]   |
| 7   | IRL and LLM RL                    | [slides][sec7]   |
| 8   | Model-Based RL                    | [slides][sec8]   |
| 9   | Offline RL                        | [slides][sec9]   |

## Homework

| Homework | Topic                       | Handout        | Code         |
| -------- | --------------------------- | -------------- | ------------ |
| HW1      | Imitation Learning          | [handout][hw1] | [code][hw1c] |
| HW2      | Policy Gradients            | [handout][hw2] | [code][hw2c] |
| HW3      | Q-Learning and Actor Critic | [handout][hw3] | [code][hw3c] |
| HW4      | LLM RL                      | [handout][hw4] | [code][hw4c] |
| HW5      | Offline RL                  | [handout][hw5] | [code][hw5c] |

## Final Project

| Document                             | Handout   | Code        |
| ------------------------------------ | --------- | ----------- |
| Final Project Outline                | [pdf][p1] | —           |
| Offline-to-Online RL Default Project | [pdf][p2] | [code][p2c] |
| LLM RL Default Project               | [pdf][p3] | [code][p3c] |

[l01]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-1.pdf
[l02]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-2.pdf
[l03]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-3.pdf
[l04]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf
[l05]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf
[l06]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf
[l07]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-7.pdf
[l08]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-8.pdf
[l09]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-9.pdf
[l10]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf
[l11]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-11.pdf
[l12]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-12.pdf
[l13]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-13.pdf
[l14]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-14.pdf
[l15]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-15.pdf
[l16]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-16.pdf
[l17]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-17.pdf
[l18]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-18.pdf
[l19]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-19.pdf
[l20]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-20.pdf
[l21]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-21.pdf
[l22]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-22.pdf
[l23]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-23.pdf
[l24]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-24.pdf
[l25]: https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-25.pdf
[sec1]: https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-1.pdf
[sec2-1]: https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-2-1.pdf
[sec2-2]: https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-2-2.pdf
[sec3]: https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-3.pdf
[sec4]: https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-4.pdf
[sec5]: https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-5.pdf
[sec6]: https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-6.pdf
[sec7]: https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-7.pdf
[sec8]: https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-8.pdf
[sec9]: https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-9.pdf
[hw1]: Homework/Handouts/hw1.pdf
[hw1c]: Homework/homework_spring2026/hw1/
[hw2]: Homework/Handouts/hw2.pdf
[hw2c]: Homework/homework_spring2026/hw2/
[hw3]: Homework/Handouts/hw3.pdf
[hw3c]: Homework/homework_spring2026/hw3/
[hw4]: Homework/Handouts/hw4.pdf
[hw4c]: Homework/homework_spring2026/hw4/
[hw5]: Homework/Handouts/hw5.pdf
[hw5c]: Homework/homework_spring2026/hw5/
[p1]: Project/final_project_outline.pdf
[p2]: Project/offline_to_online_rl_default_final_project.pdf
[p2c]: Homework/homework_spring2026/final_project_offline_online/
[p3]: Project/llm_rl_default_final_project.pdf
[p3c]: Homework/homework_spring2026/final_project_llm_rl/
[hw-repo]: https://github.com/berkeleydeeprlcourse/homework_spring2026
