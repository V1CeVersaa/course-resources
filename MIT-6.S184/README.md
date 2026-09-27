# MIT 6.S184: Generative AI with Stochastic Differential Equations

Slides and labs follow IAP 2026. Material that only exists in 2025 is kept in `Archive-2025/`.

- 2026: https://diffusion.csail.mit.edu/2026/index.html
- 2025: https://diffusion.csail.mit.edu/2025/index.html
- Lecture notes: [Introduction to Flow Matching and Diffusion Models][notes] ([arXiv][arxiv])

## Layout

| Path              | Content                                                      | In git |
| ----------------- | ------------------------------------------------------------ | ------ |
| `Slides/`         | 2026 lecture slides                                          | No     |
| `Notes/`          | Official lecture notes                                       | No     |
| `Archive-2025/`   | 2025 slides with no 2026 counterpart                         | No     |
| `Labs/`           | My lab notebooks, the animation from lab 1, upstream LICENSE | Yes    |
| `Labs/Solutions/` | Official solutions for labs 1 and 2                          | Yes    |

Links below point to the official site. Local copies use the same file names.

## Lectures (2026)

| #   | Topic                                          | Slides       |
| --- | ---------------------------------------------- | ------------ |
| 1   | Flow and Diffusion Models                      | [slides][s1] |
| 2   | Flow Matching                                  | [slides][s2] |
| 3-A | Score Functions and Score Matching             | [slides][s3] |
| 3-B | Classifier-free Guidance                       | [slides][s3] |
| 4   | Latent Spaces and Neural Network Architectures | [slides][s4] |
| 5   | Discrete Diffusion Models                      | [slides][s5] |

## Labs

| Lab   | Topic                            | Notebook     | Solution         |
| ----- | -------------------------------- | ------------ | ---------------- |
| Lab 1 | Working with ODEs and SDEs       | [mine][lab1] | [official][sol1] |
| Lab 2 | Flow Matching and Score Matching | [mine][lab2] | [official][sol2] |
| Lab 3 | Diffusion Transformer and VAEs   | [mine][lab3] | [official][sol3] |

Starter code and official solutions come from [eje24/iap-diffusion-labs][labs-repo] under its MIT License, copied to `Labs/LICENSE`. `Labs/lab_three_no_outputs.ipynb` is my lab 3 without cell outputs.

## 2025 only

| # | Topic                                  | Slides       |
| - | -------------------------------------- | ------------ |
| 5 | Guest: Generative Robotics (Burchfiel) | —            |
| 6 | Guest: Generative Protein Design (Yim) | [slides][a6] |

[s1]: https://diffusion.csail.mit.edu/2026/docs/20260120_Lecture_01.pdf
[s2]: https://diffusion.csail.mit.edu/2026/docs/20260122_Lecture_02.pdf
[s3]: https://diffusion.csail.mit.edu/2026/docs/20260123_Lecture_03.pdf
[s4]: https://diffusion.csail.mit.edu/2026/docs/20260128_Lecture_04_edited.pdf
[s5]: https://diffusion.csail.mit.edu/2026/docs/20260130_Lecture_05.pdf
[lab1]: Labs/lab_one.ipynb
[sol1]: Labs/Solutions/lab_one_complete.ipynb
[lab2]: Labs/lab_two.ipynb
[sol2]: Labs/Solutions/lab_two_complete.ipynb
[lab3]: Labs/lab_three.ipynb
[sol3]: https://github.com/eje24/iap-diffusion-labs/blob/2026/solutions/lab_three_complete.ipynb
[a6]: https://diffusion.csail.mit.edu/2025/docs/slides_lecture_6.pdf
[notes]: https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf
[arxiv]: https://arxiv.org/abs/2506.02070
[labs-repo]: https://github.com/eje24/iap-diffusion-labs/tree/2026
