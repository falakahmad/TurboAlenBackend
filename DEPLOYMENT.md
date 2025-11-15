# Backend Deployment Guide

## 📋 Pre-Deployment Checklist

- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Environment variables configured (`.env` file)
- [ ] Google credentials placed in `config/` directory
- [ ] Data directories created and writable
- [ ] Logs directory created and writable

## 🔐 Credential Management

### Google Service Account (Recommended)

1. **Place credentials file:**
   ```
   backend/config/google_credentials.json
   ```

2. **Set environment variable:**
   ```bash
   GOOGLE_SERVICE_ACCOUNT_FILE=config/google_credentials.json
   ```

3. **Security:**
   - ✅ File is gitignored (`.gitignore`)
   - ✅ Never commit to repository
   - ✅ Use environment variables in production
   - ✅ Restrict file permissions: `chmod 600 config/google_credentials.json`

### Alternative: Environment Variables

For production, you can store credentials as environment variables:

```bash
GOOGLE_SERVICE_ACCOUNT_TYPE=service_account
GOOGLE_SERVICE_ACCOUNT_PROJECT_ID=your-project-id
GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY_ID=your-key-id
GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL=your-service-account@project.iam.gserviceaccount.com
```

**Note:** You'll need to modify `backend/utils.py` to read from env vars if using this approach.

## 🚀 Deployment Options

### Option 1: Direct Python

```bash
cd backend
python -m backend.api.main
```

### Option 2: Uvicorn

```bash
cd backend
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

### Option 3: Gunicorn + Uvicorn Workers

```bash
cd backend
gunicorn backend.api.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### Option 4: Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Set environment
ENV PYTHONPATH=/app
ENV REFINER_OUTPUT_DIR=/app/data/output

# Create data directories
RUN mkdir -p data/file_versions data/strategy_feedback data/output logs templates config

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t turbo-alan-backend .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -v $(pwd)/backend/config:/app/config \
  turbo-alan-backend
```

## 🌐 Production Considerations

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # SSE specific headers
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header X-Accel-Buffering no;
    }
}
```

### Systemd Service

Create `/etc/systemd/system/turbo-alan-backend.service`:

```ini
[Unit]
Description=Turbo Alan Backend API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/backend
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable turbo-alan-backend
sudo systemctl start turbo-alan-backend
```

## 📊 Monitoring

- Logs: `backend/logs/refiner.log`
- Health check: `GET /health`
- Metrics: `GET /analytics/summary`

## 🔧 Troubleshooting

### Permission Issues
```bash
chmod -R 755 backend/data backend/logs
chown -R www-data:www-data backend/data backend/logs
```

### Port Already in Use
```bash
# Find process
lsof -i :8000
# Kill process
kill -9 <PID>
```

### Import Errors
```bash
# Ensure PYTHONPATH includes backend directory
export PYTHONPATH=/path/to/project:$PYTHONPATH
```

## ✅ Verification

After deployment, verify:

1. Health endpoint: `curl http://localhost:8000/health`
2. API docs: `http://localhost:8000/docs`
3. Logs: Check `backend/logs/refiner.log`

