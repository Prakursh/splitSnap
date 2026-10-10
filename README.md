# SplitSnap 🧾

SplitSnap is an AI-powered receipt extraction and bill-splitting app.

It uses a fine-tuned **Donut** model to convert receipt images into structured bill data, lets the user correct extraction mistakes, assign items to people, and fairly split taxes, service charges, and discounts.

## Features

- Receipt image upload
- Fine-tuned Donut receipt extraction
- Extracts:
  - item names
  - quantities
  - prices
  - subtotal
  - taxes / fees / discounts
  - grand total
- Manual correction UI
- Automatic bill consistency checking
- Shared-item assignment
- Proportional tax / fee / discount splitting
- Per-person explanation
- Manual final override

## Workflow

```text
Receipt Image
   ↓
Fine-tuned Donut
   ↓
Structured Bill JSON
   ↓
Manual Correction
   ↓
Consistency Check
   ↓
Add People + Assign Items
   ↓
Proportional Split
   ↓
Final Totals + Explanation
```

## Model

Base model:

`naver-clova-ix/donut-base`

Fine-tuning was done in two stages:

1. **SROIE** — auxiliary receipt understanding
2. **CORD** — training on the final SplitSnap schema

A second CORD epoch oversampled receipts containing taxes, service charges, and discounts because the first model had low charge recall.

### Final Test Results

Evaluated on **94 untouched CORD test receipts**.

| Metric | Result |
|---|---:|
| Valid JSON | 93.62% |
| Exact grand total | 91.49% |
| Exact subtotal | 85.11% |
| Exact charge set | 85.11% |
| Charge precision | 83.33% |
| Charge recall | 86.96% |
| Charge F1 | 85.11% |
| Exact item-name list | 62.77% |

Charge-focused fine-tuning improved validation charge F1 from **32.65% to 91.26%**.

## Model Checkpoint

The trained SplitSnap Donut checkpoint is hosted on Hugging Face:

`prakursh/splitsnap-donut`

The repository ID is also stored in:

```text
hf_model_repo.txt
```

When the app starts, Hugging Face Transformers automatically downloads the model.

No manual model download is required.

> The first launch may take longer because the checkpoint is approximately 758 MB.

## Bill Splitting

Shared items are divided equally.

Bill-level charges are distributed proportionally according to each person's share of the consumed items.

Example:

```text
Alice items = ₹700
Bob items   = ₹300
Tax         = ₹100

Alice gets 70% of tax = ₹70
Bob gets 30% of tax   = ₹30

Final:
Alice = ₹770
Bob   = ₹330
```

Discounts are handled in the same way using negative values.

## Consistency Check

SplitSnap checks whether:

```text
subtotal + charges ≈ grand total
```

If the values do not match, the app shows the unexplained amount and warns about a possible missing:

- discount
- tax
- service charge
- other adjustment

This prevents an incorrect extraction from silently producing an incorrect split.

## Installation

Python 3.12 recommended.

```powershell
py -3.12 -m venv .venv

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run

```powershell
python -m streamlit run app.py
```

On the first launch, the trained Donut model is downloaded automatically from Hugging Face.

## Project Structure

```text
splitSnap/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── hf_model_repo.txt
├── upload_model.py
├── splitsnap_metrics.json
└── SplitSnap_presentation.pptx
```

## Limitations

The model was mainly trained on CORD and SROIE receipts, so unusual real-world layouts, currencies, fonts, languages, or poor image quality may reduce extraction quality.

For this reason, SplitSnap treats AI output as a draft and provides a manual correction step before bill splitting.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Donut
- Streamlit
- Pandas
- CORD
- SROIE

## Future Improvements

- Fine-tune on Indian receipt datasets
- Blur and perspective augmentation
- Confidence estimation
- Better JSON repair
- Out-of-distribution detection
- Mobile-friendly UI