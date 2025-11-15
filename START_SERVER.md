# 🚀 How to Start the Backend Server

## Quick Start (For Testing)

### Step 1: Navigate to Backend Directory
```powershell
cd backend
```

### Step 2: Create Virtual Environment (if not already created)
```powershell
python -m venv venv
```

### Step 3: Activate Virtual Environment
```powershell
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Install Dependencies (if not already installed)
```powershell
pip install -r requirements.txt
```

### Step 5: Verify Environment Variables
Make sure you have a `.env` file in the `backend/` directory with at least:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 6: Start the Server

**Option A: Direct Python (Recommended for Development)**
```powershell
python -m backend.api.main
```

**Option B: Using Uvicorn Directly**
```powershell
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option C: From Project Root**
```powershell
# From project root directory
cd ..
python -m backend.api.main
```

## ✅ Success Indicators

When the server starts successfully, you should see:
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 🌐 Access Points

- **API Base URL**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)
- **Health Check**: `http://localhost:8000/health`

## 📋 Common Commands

### Check if server is running:
```powershell
# Test health endpoint
curl http://localhost:8000/health
# Or in PowerShell:
Invoke-WebRequest -Uri http://localhost:8000/health
```

### Stop the server:
Press `Ctrl+C` in the terminal where the server is running

## 🔧 Troubleshooting

### Issue: "Module not found" errors
**Solution**: Make sure you're in the backend directory and virtual environment is activated:
```powershell
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: Port 8000 already in use
**Solution**: Use a different port:
```powershell
uvicorn backend.api.main:app --host 0.0.0.0 --port 8001 --reload
```
Then update your Frontend `.env.local`:
```env
REFINER_BACKEND_URL=http://localhost:8001
NEXT_PUBLIC_REFINER_BACKEND_URL=http://localhost:8001
```

### Issue: "OPENAI_API_KEY not found"
**Solution**: Create `.env` file in `backend/` directory:
```powershell
# Copy from example
Copy-Item env.example .env
# Then edit .env and add your actual API key
```

### Issue: Permission errors
**Solution**: Make sure data directories exist:
```powershell
New-Item -ItemType Directory -Force -Path data\file_versions
New-Item -ItemType Directory -Force -Path data\strategy_feedback
New-Item -ItemType Directory -Force -Path data\output
New-Item -ItemType Directory -Force -Path logs
```

## 📝 Testing the Backend

### 1. Health Check
```powershell
Invoke-WebRequest -Uri http://localhost:8000/health
```

### 2. View API Documentation
Open in browser: `http://localhost:8000/docs`

### 3. Test with Frontend
Once backend is running, your Frontend should be able to connect to it at `http://localhost:8000`

## 🎯 Quick Reference

| Command | Description |
|---------|-------------|
| `python -m backend.api.main` | Start server (development) |
| `uvicorn backend.api.main:app --reload` | Start with auto-reload |
| `uvicorn backend.api.main:app --host 0.0.0.0 --port 8000` | Start on specific host/port |
| `pip install -r requirements.txt` | Install dependencies |
| `.\venv\Scripts\Activate.ps1` | Activate virtual environment |

## 🔗 Related Files

- **Main Application**: `backend/api/main.py`
- **Environment Template**: `backend/env.example`
- **Dependencies**: `backend/requirements.txt`
- **Full Documentation**: `backend/README.md`

