# ExpAgent

An AI-powered experiment management framework for data science competitions. Built for [Claude Code](https://claude.ai/claude-code) with skills that automate the full experiment lifecycle.

Designed primarily for Kaggle competitions (including Code Competitions), but adaptable to other data science contest platforms.

## Experiment Workflow

| Phase      | Action                                            |
| ---------- | ------------------------------------------------- |
| Understand | Review competition docs (`backlog doc list`)      |
| Plan       | Review past experiments, create backlog task      |
| Create     | `task new-exp EXP=expXXX` — scaffold experiment   |
| Implement  | Write train.py, settings.py, inference.py         |
| Train      | `task train-local` or `task train-vertex` (GCP)   |
| Record     | Update backlog task with CV / LB scores           |
| Submit     | `task submit-kaggle EXP=expXXX` (Kaggle pipeline) |

```
models/
└── exp001/
    ├── train.py           # Training script (tyro CLI)
    ├── settings.py        # Config & DirectorySettings
    ├── inference.py       # Submission pipeline
    ├── artifacts/         # Trained model weights, OOF predictions
    └── submission/        # Kaggle kernel files (optional: KAGGLE_CODE_SUB=true)
```

## Prerequisites

- [Claude Code](https://claude.ai/claude-code) — AI coding agent
- [uv](https://github.com/astral-sh/uv) — Python package manager
- [Task](https://taskfile.dev/) — Task runner (`Taskfile.yml`)
- [yq](https://github.com/mikefarah/yq) — YAML processor (`brew install yq`)
- [Terraform](https://github.com/hashicorp/terraform) — Infrastructure provisioning
- [Backlog](https://github.com/MrLesk/Backlog.md) — Project management CLI
- [Playwright CLI](https://github.com/microsoft/playwright-cli) — Browser automation
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) — Google Cloud SDK

## Setup

### 1. Configure project settings

**`project.yml`** (git-tracked) — Competition settings and metadata:

```yaml
competition_name: "my-competition"
competition_platform: kaggle # kaggle, signate, etc.
is_code_competition: false

metadata:
  url: "https://www.kaggle.com/competitions/my-competition"
  # Add any key-value pairs: deadline, team_size_limit, etc.
```

**`.env`** (git-ignored) — Secrets and GCP settings. Copy from `.env.example`:

```bash
cp .env.example .env
```

```env
# Kaggle authentication
KAGGLE_USERNAME=
KAGGLE_API_TOKEN=
# KAGGLE_KEY=        # Optional legacy API key when not using KAGGLE_API_TOKEN

# Google Cloud settings
PROJECT_ID=
BUCKET_NAME=        # defaults to competition_name if empty
REGION=

# Experiment tracking
MLFLOW_TRACKING_URI=http://localhost:5000
```

If you set `MLFLOW_TRACKING_URI`, make sure an MLflow tracking server is running at that address before executing training commands.
Leave `MLFLOW_TRACKING_URI` empty to disable MLflow logging for local experimentation.

### 2. Install third-party Claude Code skills

```bash
npx skills add microsoft/playwright-cli --agent claude-code --skill playwright-cli --yes
```

MLflow support is bundled in `.claude/skills/mlflow-primary/`.
