# Backend (FastAPI) - Deployment Ready

FastAPI service powering multi-pass refinement, real-time progress (SSE/WS), diffs, analytics, and job management.

## 🏗️ Structure

```
backend/
├── api/              # FastAPI application and routes
├── core/             # Core functionality (database, file versions, etc.)
├── config/           # Configuration files (heuristics.yaml, credentials)
├── data/             # Data storage (file_versions, strategy_feedback, output)
├── logs/             # Application logs
├── templates/        # Style templates (.docx files)
├── requirements.txt  # Python dependencies
└── env.example       # Environment variables template
```

## 📦 Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

### 1. Environment Variables

Copy `env.example` to `.env` and configure:

```bash
cp env.example .env
```

**Required:**
- `OPENAI_API_KEY` - Your OpenAI API key

**Optional:**
- `GOOGLE_SERVICE_ACCOUNT_FILE` - Path to Google service account JSON (default: `config/google_credentials.json`)
- `REFINER_OUTPUT_DIR` - Output directory (default: `data/output`)
- `BACKEND_API_KEY` - Optional API key for endpoint protection

### 2. Google Drive Credentials

**Option 1: Service Account (Recommended for Production)**
1. Place your service account JSON file at `config/google_credentials.json`
2. Set `GOOGLE_SERVICE_ACCOUNT_FILE=config/google_credentials.json` in `.env`

**Option 2: OAuth (For Development)**
1. Place OAuth credentials at `config/credentials.json`
2. Set `GOOGLE_CREDENTIALS_FILE=config/credentials.json` in `.env`
3. Token will be stored at `config/token.json` after first auth

**⚠️ Security Note:** Never commit credential files to git! They are in `.gitignore`.

### 3. Configuration Files

- `config/heuristics.yaml` - Refinement heuristics and settings (already included)

## 🚀 Running

### Development

```bash
python -m backend.api.main
```

Default: `http://0.0.0.0:8000`

### Production (with Uvicorn)

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Production (with Gunicorn)

```bash
gunicorn backend.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📡 Key Endpoints

- `POST /refine/run` - Start refinement; streams progress via SSE
- `GET /ws/progress/{job_id}` - WebSocket broadcast for job events
- `GET /jobs` - List all jobs
- `GET /jobs/{job_id}/status` - Get job status
- `GET /analytics/summary` - Usage and job metrics
- `GET /refine/diff?fileId=...&fromPass=...&toPass=...` - Diff across passes
- `GET /style/templates` - List .docx style templates
- `POST /strategy/feedback` - Record strategy feedback
- `GET /health` - Health check

## 📁 Data Directories

All data is stored within the `backend/` directory:

- `data/file_versions/` - File version history for diffs
- `data/strategy_feedback/` - User strategy feedback
- `data/output/` - Refined file outputs
- `logs/` - Application logs
- `templates/` - Style templates (.docx files)

## 🔒 Security

1. **Credentials**: Store Google credentials in `config/` directory (gitignored)
2. **API Key**: Optionally protect endpoints with `BACKEND_API_KEY`
3. **File Paths**: All paths are sanitized and restricted to backend directory
4. **Environment Variables**: Never commit `.env` file

## 🐳 Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

ENV PYTHONPATH=/app
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔧 Troubleshooting

### Logs Location
- Logs are written to `backend/logs/refiner.log`
- Set `DEBUG=true` in `.env` for console output

### File Permissions
- Ensure `data/` and `logs/` directories are writable
- Check `REFINER_OUTPUT_DIR` has write permissions

### Google Drive Issues
- Verify credentials file exists at configured path
- Check service account has Drive API enabled
- Ensure OAuth scopes are correct

## 📝 Notes

- All paths are relative to `backend/` directory
- Configuration files are in `config/`
- Data files are in `data/`
- The backend is self-contained and deployment-ready
