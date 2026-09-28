# AI Meeting Notes Generator

An end-to-end speech-to-notes pipeline: it takes a meeting as audio, video or raw transcript
and returns a structured summary with speaker-attributed action items, decisions and
deadlines.

## Results

Fine-tuned BART-samsum reaches **79.05% ROUGE-1 recall** on the AMI Meeting Corpus,
outperforming the FLAN-T5 baseline by **7.83 percentage points**.

Recall is the metric that matters here rather than precision or F1: in meeting notes, the
expensive failure is dropping a decision someone has to act on, not phrasing it more verbosely
than the reference summary does.

## How it works

```
audio / video ──▶ Whisper ASR ──▶ transcript ──▶ domain router ──▶ BART-samsum ──▶ summary
                                                                              └──▶ action items
                                                                                   decisions
                                                                                   deadlines
```

| Stage | Model |
| --- | --- |
| Speech recognition | OpenAI Whisper |
| Summarisation | BART-samsum, fine-tuned on AMI |
| Baseline for comparison | FLAN-T5 |
| Interface | Gradio |

## Data

Trained and evaluated across three meeting domains so the summariser is not tuned to a single
speaking style:

- **Business** — [AMI Meeting Corpus](https://huggingface.co/datasets/edinburghcstr/ami)
  (utterance-level) paired with [AMI abstractive summaries](https://huggingface.co/datasets/knkarthick/AMI)
- **Education** — ICSI Meeting Corpus
- **Healthcare** — medical consultation transcripts

## Running it

The full pipeline is in [`final-poject-ai-2__1_.ipynb`](final-poject-ai-2__1_.ipynb), written
for a GPU Kaggle kernel.

```bash
pip install openai-whisper transformers rouge-score nltk sentencepiece librosa soundfile gradio
```

The notebook sets up its own working tree under `meeting_intelligence/` — raw data, processed
splits, fine-tuned checkpoints and the Gradio demo assets — and runs end to end from dataset
download through to the launched demo.

A Hugging Face token is required for the gated AMI summaries dataset.

## Why fine-tune rather than prompt

Meeting transcripts break most summarisers: they are long, disfluent, heavily interrupted, and
the important content is thinly distributed across a lot of filler. Fine-tuning BART on
AMI-style dialogue teaches it that structure directly, which is where the 7.83-point gain over
the FLAN-T5 baseline comes from.

---

Built by [Siddhi Kakani](https://github.com/siddhi1703) · M.S. Artificial Intelligence,
Northeastern University
