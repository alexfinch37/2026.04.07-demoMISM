# 2026.04.07-demoMISM

#DemoMISM

A demo project for MISM coursework, built with Python 3.12.

## Setup

```bash
# Clone the repo
git clone https://github.com/alexfinch37/2026.04.07-demoMISM.git
cd 2026.04.07-demoMISM

# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

## Project Structure

```
DemoMISM/
├── .gitignore
├── README.md
├── requirements.txt
└── ...
```

## Dependencies

Key packages include:
- **Polars** — fast DataFrame library
- **PyTorch** — machine learning framework
- **Flask / Werkzeug** — web framework
- **PyYAML** — YAML parsing

See `requirements.txt` for the full list.

## Notes

- `.venv` is excluded from version control. Run `pip install -r requirements.txt` after cloning to restore your environment.
- Python 3.12 required.
