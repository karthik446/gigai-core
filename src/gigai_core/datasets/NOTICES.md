# Eval dataset notices

The loaders in this package read public datasets. Each loader's `LOADER.sources` lists the upstream files and their
pinned sha256; each slice row keeps the upstream ids in `source_ids`. Items are reformatted into one prompt string
(MT-Bench: one scripted user turn per conversation turn); their text is not altered.

Six sets ship their `ci` slice inside the wheel (`data/<set>.jsonl` with a `.sha256` sidecar); four are download-only:
`build_release(name, directory)` fetches the upstream files onto the user's machine and builds the slice there.

| Set | SPDX | In the wheel | Why |
| --- | --- | --- | --- |
| contractnli | CC-BY-4.0 | `ci` slice | attribution carried here |
| ragtruth | MIT | `ci` slice | attribution carried here |
| ifeval | Apache-2.0 | `ci` slice | attribution carried here |
| jsonschemabench | MIT | `ci` slice | attribution carried here |
| tinygsm8k | MIT | `ci` slice | attribution carried here |
| tinymmlu | MIT | `ci` slice | attribution carried here |
| quality | CC-BY-4.0 | download-only | 242 KB of article text under per-article licences (Gutenberg, OANC, others) |
| rewardbench2 | ODC-By-1.0 | download-only | the responses are third-party model outputs under their own terms |
| judgebench | MIT | download-only | the responses are GPT-4o outputs, subject to OpenAI's terms of use |
| mtbench_human | CC-BY-4.0 | download-only | the answers are third-party model outputs under their own terms |

## contractnli

- Name: ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts
- SPDX: CC-BY-4.0
- Licence: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- URL: https://stanfordnlp.github.io/contract-nli/ (file `contract-nli.zip`, test split)
- Fetched: 2026-11-10
- Attribution: Copyright Hitachi America, Ltd., made available under the Creative Commons Attribution 4.0
  International Public License (the `LICENSE` and `TERMS` files in the archive). Yuta Koreeda and Christopher D.
  Manning, "ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts", Findings of EMNLP 2021.
- Changes made: sliced and flattened. A seeded subset of test documents is kept; each document and its 17
  hypotheses are rendered as one prompt; span indices become character offsets.

## ragtruth

- Name: RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models
- SPDX: MIT
- URL: https://github.com/ParticleMedia/RAGTruth (files `dataset/response.jsonl`, `dataset/source_info.jsonl`)
- Fetched: 2026-11-10
- Attribution: Cheng Niu et al., ACL 2024. The source passages come from MS MARCO, CNN/Daily Mail and Yelp as
  redistributed by RAGTruth.
- Changes made: sliced and flattened. A seeded subset of test responses is kept; the task prompt and the response
  are joined into one prompt; `implicit_true` spans are dropped.
- Licence (the repository's `LICENSE`, verbatim):

  ```text
  MIT License

  Copyright (c) 2023 Particle Media

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```

## quality

- Name: QuALITY: Question Answering with Long Input Texts, Yes! (v1.0.1, `htmlstripped` dev split)
- SPDX: CC-BY-4.0
- Licence: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- URL: https://github.com/nyu-mll/quality (file `data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev`); licence stated at
  https://nyu-mll.github.io/quality/
- Fetched: 2026-11-10
- Attribution: NYU Machine Learning for Language, CC BY 4.0. Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi,
  Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He and Samuel R.
  Bowman, NAACL 2022. Each article keeps its own source licence in the upstream `license` field (Project Gutenberg
  licence, the Open American National Corpus licence, and others).
- Changes made: sliced and flattened. A seeded subset of articles and questions is kept; article, question and the
  four options are rendered as one prompt; the gold label becomes a letter.

## ifeval

- Name: IFEval: Instruction-Following Evaluation for Large Language Models
- SPDX: Apache-2.0
- URL: https://huggingface.co/datasets/google/IFEval at revision 966cd89545d6b6acfd7638bc708b98261ca58e84 (file
  `ifeval_input_data.jsonl`)
- Fetched: 2026-11-10
- Attribution: Copyright Google LLC (the dataset card's `license: apache-2.0`). Jeffrey Zhou, Tianjian Lu, Swaroop
  Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou and Le Hou, arXiv:2311.07911, 2023.
- Changes made: sliced. A seeded subset of prompts is kept unchanged.
- Licence notice:

  ```text
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use these files except in compliance with the License.
  You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ```

## jsonschemabench

- Name: JSONSchemaBench (`Github_medium` subset, test split)
- SPDX: MIT
- URL: https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench at revision 5bd0f4640badc6f3f02df796421d21cb0ca0b141
  (file `Github_medium/test-00000-of-00001.parquet`)
- Fetched: 2026-11-11
- Attribution: Saibo Geng et al., "Generating Structured Outputs from Language Models: Benchmark and Studies",
  arXiv:2501.10868, 2025. The dataset card says "This dataset is provided under the [MIT License]
  (https://opensource.org/licenses/MIT)"; the repository has no `LICENSE` file, so the copyright line below names
  the authors as the card's citation does. The card lists the schemas' sources as GitHub, Kubernetes, API
  specifications and curated collections.
- Changes made: sliced. A seeded subset of schemas is kept unchanged as each item's expected value; one fixed
  instruction is the prompt.
- Licence (MIT, the text at https://opensource.org/licenses/MIT):

  ```text
  MIT License

  Copyright (c) 2025 Saibo Geng et al. (the JSONSchemaBench authors)

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```

## rewardbench2

- Name: RewardBench 2 (`Focus` and `Factuality` subsets, test split)
- SPDX: ODC-By-1.0
- Licence: Open Data Commons Attribution License v1.0, https://opendatacommons.org/licenses/by/1-0/
- URL: https://huggingface.co/datasets/allenai/reward-bench-2 at revision 7ff08853b0d5686e79b13fda8677024f566a104a
  (file `data/test-00000-of-00001.parquet`)
- Fetched: 2026-11-11
- Attribution: Contains information from RewardBench 2, which is made available here under the Open Data Commons
  Attribution License v1.0 (https://opendatacommons.org/licenses/by/1-0/). Allen Institute for AI. Saumya Malik,
  Valentina Pyatkin, Sander Land, Jacob Morrison, Noah A. Smith, Hannaneh Hajishirzi and Nathan Lambert, "RewardBench
  2: Advancing Reward Model Evaluation", arXiv:2506.01937, 2025. The card adds: "This dataset includes output data
  generated from third party models that are subject to separate terms governing their use"; it lists each model's
  terms (among them OpenAI's terms of use for GPT-4o outputs, Anthropic's terms of service and usage policy for
  Claude 3.5 Sonnet outputs, the Llama community licences, the Qwen licence and the DeepSeek model licence's use
  restrictions). Each item records which model wrote each response in `source_ids` (`model_A` to `model_D`).
- Changes made: sliced and flattened. A seeded subset of prompts is kept; each prompt and its four responses are
  rendered as one prompt with the responses in a seeded order, lettered A-D; the chosen response becomes a letter;
  item ids carry the subset because upstream ids repeat across subsets. The dataset card also asks for research and
  educational use in line with Ai2's Responsible Use Guidelines (guidance, not a licence term).

## judgebench

- Name: JudgeBench: A Benchmark for Evaluating LLM-Based Judges (`gpt` split)
- SPDX: MIT
- URL: https://huggingface.co/datasets/ScalerLab/JudgeBench at revision 57dd5e0b9817d07f05ec8f45a91b2ce1e310e308
  (file `data/gpt-00000-of-00001.jsonl`)
- Fetched: 2026-11-11
- Attribution: Sijun Tan, Siyuan Zhuang, Kyle Montgomery, Willian Yuan Tang, Alejandro Cuadron, Chenguang Wang,
  Raluca Ada Popa and Ion Stoica, arXiv:2410.12784, 2024. The dataset card states `license: mit`; neither the card
  nor the GitHub repository carries a `LICENSE` file, so the copyright line below names the authors as the card's
  citation does. The questions come from MMLU-Pro, LiveBench and LiveCodeBench; both responses of a pair are
  GPT-4o (2024-05-13) outputs, subject to OpenAI's terms of use.
- Changes made: sliced and flattened. A seeded subset of pairs is kept; question and both responses are rendered as
  one prompt, with A and B swapped by a seeded coin per pair (the swap and the source label are kept in
  `source_ids`).
- Licence (MIT, the text at https://opensource.org/licenses/MIT):

  ```text
  MIT License

  Copyright (c) 2024 Sijun Tan, Siyuan Zhuang, Kyle Montgomery, Willian Yuan Tang,
  Alejandro Cuadron, Chenguang Wang, Raluca Ada Popa and Ion Stoica

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```

## mtbench_human

- Name: MT-Bench human judgments (`human` split)
- SPDX: CC-BY-4.0
- Licence: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- URL: https://huggingface.co/datasets/lmsys/mt_bench_human_judgments at revision
  f7d2896d2cc5d80f8b55c2bbc722613555233c25 (file `data/human-00000-of-00001-25f4910818759289.parquet`)
- Fetched: 2026-11-11
- Attribution: LMSYS, CC BY 4.0. Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao
  Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez and Ion Stoica, "Judging
  LLM-as-a-judge with MT-Bench and Chatbot Arena", arXiv:2306.05685, 2023. The kept answers are outputs of GPT-4,
  GPT-3.5-turbo, Vicuna-13B, Alpaca-13B and LLaMA-13B.
- Changes made: filtered, sliced and flattened. Judgments are grouped per question, turn and ordered model pair;
  only groups with two or more judgments that agree on a non-tie winner and have no Claude model are kept; the two
  conversations are rendered as scripted user turns, one per exchange (the user's question with both assistants'
  answers), with the judging instruction as the system prompt; the winner becomes a letter.

## tinygsm8k

- Name: tinyGSM8k (tinyBenchmarks), a 100-problem subset of GSM8K's test split
- SPDX: MIT
- URL: https://huggingface.co/datasets/tinyBenchmarks/tinyGSM8k at revision b870449ed96ee3e308748b550908332450ec4098
  (file `main/test-00000-of-00001.parquet`); licences from
  https://github.com/felipemaiapolo/tinyBenchmarks/blob/deabaab9d654f3bb0989cc8672ebe3aacfb8e9a1/LICENSE and
  https://github.com/openai/grade-school-math/blob/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/LICENSE
- Fetched: 2026-11-11
- Attribution: Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu and Mikhail Yurochkin,
  "tinyBenchmarks: evaluating LLMs with fewer examples", arXiv:2402.14992, 2024. GSM8K: Karl Cobbe et al.,
  "Training Verifiers to Solve Math Word Problems", arXiv:2110.14168, 2021. The Hugging Face card carries no
  licence tag; the tinyBenchmarks repository and GSM8K are both MIT, so both licences are carried below.
- Changes made: sliced. A seeded subset of problems is kept; the question text is unchanged in the prompt, after a
  fixed answer-format instruction; the gold answer is the number after `####`, commas dropped.
- Licence (the two repositories' `LICENSE` files, identical except for the copyright line, which both carry here):

  ```text
  MIT License

  Copyright (c) 2024 Felipe Maia Polo
  Copyright (c) 2021 OpenAI

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```

## tinymmlu

- Name: tinyMMLU (tinyBenchmarks), a 100-question subset of MMLU's test split
- SPDX: MIT
- URL: https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU at revision cd6a8ee9e1e6f3f18f683535f6c096c386a85da3
  (file `all/test-00000-of-00001.parquet`); licences from
  https://github.com/felipemaiapolo/tinyBenchmarks/blob/deabaab9d654f3bb0989cc8672ebe3aacfb8e9a1/LICENSE and
  https://github.com/hendrycks/test/blob/0aff600098ffa002d8e9f1ffe4d6e05475edd28e/LICENSE
- Fetched: 2026-11-11
- Attribution: Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu and Mikhail Yurochkin,
  "tinyBenchmarks: evaluating LLMs with fewer examples", arXiv:2402.14992, 2024. MMLU: Dan Hendrycks et al.,
  "Measuring Massive Multitask Language Understanding", ICLR 2021. The Hugging Face card carries no licence tag;
  the tinyBenchmarks repository and MMLU are both MIT, so both licences are carried below.
- Changes made: sliced and flattened. A seeded subset of questions is kept; question and choices are rendered as one
  prompt with the choices lettered A-D; the 0-based answer index becomes a letter.
- Licence (the two repositories' `LICENSE` files, identical except for the copyright line, which both carry here):

  ```text
  MIT License

  Copyright (c) 2024 Felipe Maia Polo
  Copyright (c) 2020 Dan Hendrycks

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```

