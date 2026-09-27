# Stanford CS231N: Deep Learning for Computer Vision

I take this course as a supplement to [UMich EECS498](../UMich-EECS498/README.md) and do not plan to do all of its assignments.

Slides follow Spring 2026. Spring 2025 slides and section slides are kept in `Archive-2025/`.

- Spring 2026: https://cs231n.stanford.edu/
- Spring 2025: https://cs231n.stanford.edu/2025/
- Course notes: https://cs231n.github.io/
- Lecture videos: [Spring 2025 on YouTube][videos]. Spring 2026 videos are only on Canvas.

## Layout

| Path            | Content                                | In git |
| --------------- | -------------------------------------- | ------ |
| `Slides/`       | Spring 2026 lecture slides             | No     |
| `Sections/`     | Spring 2026 review session slides      | No     |
| `Archive-2025/` | Spring 2025 lecture and section slides | No     |

Links below point to the official site. Local copies use the same file names.

## Lectures (Spring 2026)

| #  | Topic                                   | Slides         | 2025           |
| -- | --------------------------------------- | -------------- | -------------- |
| 1  | Introduction (part 1)                   | [slides][s01a] | [slides][a01a] |
| 1  | Introduction (part 2)                   | [slides][s01b] | [slides][a01b] |
| 2  | Image Classification with Linear Models | [slides][s02]  | [slides][a02]  |
| 3  | Regularization and Optimization         | [slides][s03]  | [slides][a03]  |
| 4  | Neural Networks and Backpropagation     | [slides][s04]  | [slides][a04]  |
| 5  | Image Classification with CNNs          | [slides][s05]  | [slides][a05]  |
| 6  | CNN Architectures                       | [slides][s06]  | [slides][a06]  |
| 7  | Recurrent Neural Networks               | [slides][s07]  | [slides][a07]  |
| 8  | Attention and Transformers              | [slides][s08]  | [slides][a08]  |
| 9  | Detection, Segmentation, Visualization  | [slides][s09]  | [slides][a09]  |
| 10 | Video Understanding                     | [slides][s10]  | [slides][a10]  |
| 11 | Large Scale Distributed Training        | [slides][s11]  | [slides][a11]  |
| 12 | Self-supervised Learning                | [slides][s12]  | [slides][a12]  |
| 13 | Generative Models 1                     | [slides][s13]  | [slides][a13]  |
| 14 | Generative Models 2                     | [slides][s14]  | [slides][a14]  |
| 15 | 3D Vision                               | [slides][s15]  | [slides][a15]  |
| 16 | Vision and Language                     | [slides][s16]  | [slides][a16]  |
| 17 | Guest: World Modeling (Wetzstein)       | —              | —              |
| 18 | Human-Centered AI                       | —              | —              |

Handouts for lecture 4: [Linear backprop example][hb1], [Derivatives, backprop, and vectorization][hb2].

## Spring 2025 only

| Topic                      | Slides        |
| -------------------------- | ------------- |
| Robot Learning (Yunzhu Li) | [slides][a17] |

## Sections

| Session                | Slides       | 2025         | Colab        |
| ---------------------- | ------------ | ------------ | ------------ |
| Python / NumPy Review  | —            | —            | [colab][cpy] |
| Backprop Review        | [slides][c2] | [slides][b2] | [colab][cbp] |
| Final Project Overview | [slides][c3] | [slides][b3] |              |
| PyTorch Review         | —            | —            | [colab][cpt] |
| RNNs and Transformers  | [slides][c5] | [slides][b5] |              |
| Midterm Review         | —            | [slides][b6] |              |

## Assignments

| Assignment | Topic                                                   | Handout        |
| ---------- | ------------------------------------------------------- | -------------- |
| A1         | kNN, Softmax, Fully-Connected Nets                      | [handout][hw1] |
| A2         | BatchNorm, Dropout, CNNs, Visualization, RNN Captioning | [handout][hw2] |
| A3         | Transformer Captioning, SSL, Diffusion, CLIP and DINO   | [handout][hw3] |

The 2025 and 2026 assignments are the same apart from due dates.

[videos]: https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16
[s01a]: https://cs231n.stanford.edu/slides/2026/lecture_1_part_1.pdf
[s01b]: https://cs231n.stanford.edu/slides/2026/lecture_1_part_2.pdf
[s02]: https://cs231n.stanford.edu/slides/2026/lecture_2.pdf
[s03]: https://cs231n.stanford.edu/slides/2026/lecture_3.pdf
[s04]: https://cs231n.stanford.edu/slides/2026/lecture_4.pdf
[s05]: https://cs231n.stanford.edu/slides/2026/lecture_5.pdf
[s06]: https://cs231n.stanford.edu/slides/2026/lecture_6.pdf
[s07]: https://cs231n.stanford.edu/slides/2026/lecture_7.pdf
[s08]: https://cs231n.stanford.edu/slides/2026/lecture_8.pdf
[s09]: https://cs231n.stanford.edu/slides/2026/lecture_9.pdf
[s10]: https://cs231n.stanford.edu/slides/2026/lecture_10.pdf
[s11]: https://cs231n.stanford.edu/slides/2026/lecture_11.pdf
[s12]: https://cs231n.stanford.edu/slides/2026/lecture_12.pdf
[s13]: https://cs231n.stanford.edu/slides/2026/lecture_13.pdf
[s14]: https://cs231n.stanford.edu/slides/2026/lecture_14.pdf
[s15]: https://cs231n.stanford.edu/slides/2026/lecture_15.pdf
[s16]: https://cs231n.stanford.edu/slides/2026/lecture_16.pdf
[a01a]: https://cs231n.stanford.edu/slides/2025/lecture_1_part_1.pdf
[a01b]: https://cs231n.stanford.edu/slides/2025/lecture_1_part_2.pdf
[a02]: https://cs231n.stanford.edu/slides/2025/lecture_2.pdf
[a03]: https://cs231n.stanford.edu/slides/2025/lecture_3.pdf
[a04]: https://cs231n.stanford.edu/slides/2025/lecture_4.pdf
[a05]: https://cs231n.stanford.edu/slides/2025/lecture_5.pdf
[a06]: https://cs231n.stanford.edu/slides/2025/lecture_6.pdf
[a07]: https://cs231n.stanford.edu/slides/2025/lecture_7.pdf
[a08]: https://cs231n.stanford.edu/slides/2025/lecture_8.pdf
[a09]: https://cs231n.stanford.edu/slides/2025/lecture_9.pdf
[a10]: https://cs231n.stanford.edu/slides/2025/lecture_10.pdf
[a11]: https://cs231n.stanford.edu/slides/2025/lecture_11.pdf
[a12]: https://cs231n.stanford.edu/slides/2025/lecture_12.pdf
[a13]: https://cs231n.stanford.edu/slides/2025/lecture_13.pdf
[a14]: https://cs231n.stanford.edu/slides/2025/lecture_14.pdf
[a15]: https://cs231n.stanford.edu/slides/2025/lecture_15.pdf
[a16]: https://cs231n.stanford.edu/slides/2025/lecture_16.pdf
[a17]: https://cs231n.stanford.edu/slides/2025/lecture_17.pdf
[hb1]: https://cs231n.stanford.edu/handouts/linear-backprop.pdf
[hb2]: https://cs231n.stanford.edu/handouts/derivatives.pdf
[cpy]: https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb
[cbp]: https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/backprop.ipynb
[cpt]: https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/pytorch.ipynb
[c2]: https://cs231n.stanford.edu/slides/2026/section_2_backprop.pdf
[c3]: https://cs231n.stanford.edu/slides/2026/section_3_project.pdf
[c5]: https://cs231n.stanford.edu/slides/2026/section_5.pdf
[b2]: https://cs231n.stanford.edu/slides/2025/section_2.pdf
[b3]: https://cs231n.stanford.edu/slides/2025/section_3.pdf
[b5]: https://cs231n.stanford.edu/slides/2025/section_5.pdf
[b6]: https://cs231n.stanford.edu/slides/2025/section_6.pdf
[hw1]: https://cs231n.github.io/assignments2026/assignment1/
[hw2]: https://cs231n.github.io/assignments2026/assignment2/
[hw3]: https://cs231n.github.io/assignments2026/assignment3/
