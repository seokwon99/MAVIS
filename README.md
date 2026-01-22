# Multimodal Attribution for Visual Question Answering (MAVIS)


[**📖 MAVIS arXiv**](https://arxiv.org/abs/2511.12142) | [**🤗 QA Dataset**](https://huggingface.co/datasets/seokwon99/MAVIS) | [**🤗 Corpus Dataset**](https://huggingface.co/datasets/seokwon99/MAVIS_documents)

## News

* 🔥 [2026-01-22] MAVIS is accepted to AAAI 2026.
* 🔥 [2026-01-22] We release the MAVIS dataset on Hugging Face, including the QA set and the multimodal corpus for retrieval.  


## Dataset Summary
MAVIS is a new dataset for open-domain, long-form visual question answering, characterized by three key features: (1) the questions incorporate input images, requiring visual understanding to correctly interpret the user’s intent; (2) the desired answers are long-form, necessitating the retrieval and synthesis of diverse information rather than isolated facts; (3) each question is accompanied by gold-standard supporting multimodal documents.

We release the multimodal corpus [here](https://huggingface.co/datasets/seokwon99/MAVIS_documents).

<img src="https://huggingface.co/datasets/seokwon99/MAVIS/resolve/main/mavis_intro.png" width="400" />

## Load Dataset

```python
from datasets import load_dataset
mavis_test = load_dataset("seokwon99/MAVIS", split="test")
corpus = load_dataset("seokwon99/MAVIS_documents")
```

## Evaluation
The evaluation code will be available soon.

## Have any questions?

Please contact at seokwon.song@vision.snu.ac.kr

## Citation
Please cite our work:

```bibtex  
@article{song2025mavis,
  title={MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering},
  author={Song, Seokwon and Park, Minsu and Kim, Gunhee},
  journal={arXiv preprint arXiv:2511.12142},
  year={2025}
}
```
