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
- Automatic bill consistency check
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

## Final Test Results

Evaluated on 94 untouched CORD test receipts.

| Metric | Result |
|---|---:|
| Valid JSON | 93.62% |
| Exact grand total | 91.49% |
| Exact subtotal | 85.11% |
| Exact charge set | 85.11% |
| Charge F1 | 85.11% |
| Exact item-name list | 62.77% |

Charge-focused fine-tuning improved validation charge F1 from **32.65% → 91.26%**.

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

Discounts are handled the same way using negative values.

## Consistency Check

SplitSnap checks whether:

```text
subtotal + charges ≈ grand total
```

If the values do not match, it warns the user about a possible missing:

- discount
- tax
- service charge
- other adjustment

This prevents a bad extraction from silently producing a wrong split.

## Installation

Python 3.12 recommended.

```powershell
py -3.12 -m venv .venv

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Place the trained checkpoint in:

```text
splitsnap_model/
```

Then run:

```powershell
python -m streamlit run app.py
```

## Project Structure

```text
splitSnap/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── splitsnap_metrics.json
├── samples/
└── splitsnap_model/   # not tracked in Git
```

## Limitations

The model was mainly trained on CORD and SROIE receipts, so unusual real-world layouts, currencies, fonts, or poor image quality may reduce extraction quality.

For that reason, SplitSnap treats AI output as a draft and always allows user correction before splitting.

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
- Receipt image augmentation for blur and perspective distortion
- Confidence estimation
- Automatic JSON repair
- Better out-of-distribution detection
- Mobile-friendly interface
- Persistent bill history