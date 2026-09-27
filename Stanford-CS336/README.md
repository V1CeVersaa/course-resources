# Stanford CS336: Language Modeling from Scratch

Lectures and assignments follow Spring 2025.

- Spring 2025: https://cs336.stanford.edu/spring2025/
- Latest offering: https://cs336.stanford.edu/

## Layout

| Path           | Content                                             | In git    |
| -------------- | --------------------------------------------------- | --------- |
| `slides-sp25/` | Official lecture repo: executable lectures and PDFs | Submodule |
| `assignment*/` | Assignments, on top of the official starter repos   | Yes       |

Executable lectures open in the official trace viewer. PDF lectures link to the lecture repo, which is also checked out locally as `slides-sp25/`.

## Lectures (Spring 2025)

| #  | Topic                          | Slides       |
| -- | ------------------------------ | ------------ |
| 1  | Overview, tokenization         | [trace][l01] |
| 2  | PyTorch, resource accounting   | [trace][l02] |
| 3  | Architectures, hyperparameters | [pdf][l03]   |
| 4  | Mixture of experts             | [pdf][l04]   |
| 5  | GPUs                           | [pdf][l05]   |
| 6  | Kernels, Triton                | [trace][l06] |
| 7  | Parallelism                    | [pdf][l07]   |
| 8  | Parallelism                    | [trace][l08] |
| 9  | Scaling laws                   | [pdf][l09]   |
| 10 | Inference                      | [trace][l10] |
| 11 | Scaling laws                   | [pdf][l11]   |
| 12 | Evaluation                     | [trace][l12] |
| 13 | Data                           | [trace][l13] |
| 14 | Data                           | [trace][l14] |
| 15 | Alignment - SFT/RLHF           | [pdf][l15]   |
| 16 | Alignment - RL                 | [pdf][l16]   |
| 17 | Alignment - RL                 | [trace][l17] |
| 18 | Guest: Junyang Lin             | —            |
| 19 | Guest: Mike Lewis              | —            |

## Assignments

| Assignment | Topic                                               | Handout       | Code        |
| ---------- | --------------------------------------------------- | ------------- | ----------- |
| A1         | Basics: BPE, Transformer LM, training loop          | [handout][a1] | [code][a1c] |
| A2         | Systems: profiling, FlashAttention 2, DDP, sharding | [handout][a2] | [code][a2c] |
| A3         | Scaling laws                                        | [handout][a3] | [code][a3c] |
| A4         | Data: filtering and deduplication                   | [handout][a4] | [code][a4c] |
| A5         | Alignment: SFT, expert iteration, GRPO              | [handout][a5] | [code][a5c] |

A5 also has a [safety and RLHF supplement][a5s].

## References

- [从 0 到 1 实现 Transformer 模型-CS336 作业 1](https://www.cnblogs.com/Sanhao99/p/19057986)
- [CS336 的魔改版实现](https://github.com/eve-liya/LanguageModeling)
- [GitHub 上的某个 Assignment 1 实现](https://github.com/donglinkang2021/cs336-assignment1-basics/tree/main)
- [另一个 Assignment 1 实现，脚本和分词器不错](https://github.com/ZitongYang/cs336-assignment1-basics)

[l01]: https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_01.json
[l02]: https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_02.json
[l03]: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf
[l04]: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf
[l05]: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf
[l06]: https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_06.json
[l07]: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf
[l08]: https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_08.json
[l09]: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf
[l10]: https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_10.json
[l11]: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf
[l12]: https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_12.json
[l13]: https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_13.json
[l14]: https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_14.json
[l15]: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf
[l16]: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf
[l17]: https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_17.json
[a1]: assignment1-basics/cs336_spring2025_assignment1_basics.pdf
[a1c]: assignment1-basics/
[a2]: assignment2-systems/cs336_spring2025_assignment2_systems.pdf
[a2c]: assignment2-systems/
[a3]: assignment3-scaling/cs336_spring2025_assignment3_scaling.pdf
[a3c]: assignment3-scaling/
[a4]: assignment4-data/cs336_spring2025_assignment4_data.pdf
[a4c]: assignment4-data/
[a5]: assignment5-alignment/cs336_spring2025_assignment5_alignment.pdf
[a5c]: assignment5-alignment/
[a5s]: assignment5-alignment/cs336_spring2025_assignment5_supplement_safety_rlhf.pdf
