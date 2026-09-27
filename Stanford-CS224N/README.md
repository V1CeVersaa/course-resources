# Stanford CS224N: Natural Language Processing with Deep Learning

Slides follow Winter 2026. Material that only exists in Winter 2025 is kept in `Archive-W25/`.

- Winter 2026: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/
- Winter 2025: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/

## Layout

| Path                 | Content                                               | In git |
| -------------------- | ----------------------------------------------------- | ------ |
| `Slides/`            | Winter 2026 lecture slides                            | No     |
| `Notes/`             | Official lecture notes, shared by 2025 and 2026       | No     |
| `Archive-W25/`       | Winter 2025 slides and notes with no 2026 counterpart | No     |
| `Homework/Handouts/` | Assignment handouts                                   | Yes    |
| `Homework/a4/`       | A4 starter code, data and LaTeX template              | Yes    |

Links below point to the official site. Local copies use the same file names.

## Lectures (Winter 2026)

| #  | Topic                                             | Slides         | Notes               |
| -- | ------------------------------------------------- | -------------- | ------------------- |
| 1  | Introduction                                      | [slides][s01a] |                     |
| 1  | History of NLP                                    | [slides][s01b] |                     |
| 2  | Word Vectors                                      | [slides][s02]  | [1][n02a] [2][n02b] |
| —  | Python Review Session                             | [slides][spy]  |                     |
| 3  | Backpropagation and Neural Network Basics         | [slides][s03]  | [notes][n03]        |
| 4  | Language Models and RNNs                          | [slides][s04]  | [notes][n04]        |
| 5  | Transformers                                      | [slides][s05]  | [notes][n05]        |
| 6  | Final Projects: Custom and Default                | [slides][s06]  |                     |
| 7  | Pretraining (Scaling, Systems, Data)              | [slides][s07]  |                     |
| 8  | Post-training (RLHF, SFT, DPO)                    | [slides][s08]  |                     |
| 9  | Efficient Adaptation (Prompting + PEFT)           | [slides][s09]  |                     |
| 10 | Agents, Tool Use, and RAG                         | [slides][s10]  |                     |
| 11 | Benchmarking and Evaluation                       | [slides][s11]  |                     |
| 12 | Reasoning 1                                       | [slides][s12]  |                     |
| 13 | Reasoning 2                                       | [slides][s13]  |                     |
| 14 | Guest: Tokenization and Multilinguality (Kallini) | [slides][s14]  |                     |
| 15 | Guest: Interpretability (Been Kim)                | —              |                     |
| 16 | Social and Broader Impacts of NLP                 | [slides][s16]  |                     |
| 17 | Guest: Multimodality (Zettlemoyer)                | —              |                     |
| 18 | Guest: Tinker and LoRA Without Regret (Schulman)  | —              |                     |
| 19 | Open Questions in NLP 2026                        | [slides][s19]  |                     |

Supplementary notes for lecture 3: [Gradient notes][ng], [Review of differential calculus][nc].

## Winter 2025 only

| Topic                                      | Slides        | Notes         |
| ------------------------------------------ | ------------- | ------------- |
| Word Vectors and Language Models           | [slides][a02] |               |
| Dependency Parsing                         | [slides][a04] | [notes][an04] |
| Advanced Variants of RNNs, Attention       | [slides][a06] |               |
| Question Answering and Knowledge           | [slides][a13] |               |
| Guest: Model Analysis and Interpretability | [slides][ag]  |               |

## Homework

| Assignment | Topic                                         | Handout         | Code          |
| ---------- | --------------------------------------------- | --------------- | ------------- |
| A4 (W25)   | Self-Attention, Transformers, and Pretraining | [handout][hw4]  | [code][hw4c]  |

[s01a]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture01-intro.pdf
[s01b]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture01-history.pdf
[s02]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture02-wordvecs.pdf
[n02a]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/readings/cs224n_winter2023_lecture1_notes_draft.pdf
[n02b]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/readings/cs224n-2019-notes02-wordvecs2.pdf
[spy]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w25/2024%20CS224N%20Python%20Review%20Session%20Slides.pptx.pdf
[s03]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture03-neuralnets.pdf
[n03]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/readings/cs224n-2019-notes03-neuralnets.pdf
[s04]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture04-rnnlm.pdf
[n04]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/readings/cs224n-2019-notes05-LM_RNN.pdf
[s05]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture05-transformers.pdf
[n05]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/readings/cs224n-self-attention-transformers-2023_draft.pdf
[s06]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture06-final-project.pdf
[s07]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture07-pretraining.pdf
[s08]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture08-posttraining.pdf
[s09]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture09-peft.pdf
[s10]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture10-rag-agents.pdf
[s11]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture11-evaluation.pdf
[s12]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture12-reasoning-part1.pdf
[s13]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture13-reasoning-part2.pdf
[s14]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture14-guest-julie-tokenization-multilinguality.pdf
[s16]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture16-impact-on-humanity.pdf
[s19]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/slides_w26/cs224n-2026-lecture19-open-questions.pdf
[a02]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/slides_w25/cs224n-2025-lecture02-wordvecs2.pdf
[a04]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/slides_w25/cs224n-2025-lecture04-dep-parsing.pdf
[an04]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/readings/cs224n-2019-notes04-dependencyparsing.pdf
[a06]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/slides_w25/cs224n-2025-lecture06-fancy-rnn.pdf
[a13]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/slides_w25/cs224n-2025-lecture13-QA.pdf
[ag]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/slides_w25/cs224n-2025-guest-lecture-interpretability.pdf
[ng]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/readings/gradient-notes.pdf
[nc]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/readings/review-differential-calculus.pdf
[hw4]: Homework/Handouts/a4.pdf
[hw4c]: Homework/a4/
