# Cat Re-identification System - Production

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run deployment script
python deploy.py
```

### 2. Add Cats to Database
```bash
python production/app.py --action add --cat-id "fluffy_001" --image "path/to/cat/image.jpg"
```

### 3. Identify Cats
```bash
python production/app.py --action identify --image "path/to/unknown/cat.jpg"
```

### 4. Check Database Info
```bash
python production/app.py --action info
```

## Configuration

Edit `production/config/config.json` to customize:
- Model settings (embedding dimension, image size, threshold)
- Database paths
- Logging configuration
- Performance settings

## Docker Deployment

```bash
# Build image
docker build -t cat-reidentifier .

# Run container
docker run -v $(pwd)/production:/app/production cat-reidentifier
```

## API Usage

```python
from production.app import ProductionCatReidentifier

# Initialize
app = ProductionCatReidentifier()

# Add cat
app.add_cat("fluffy_001", "path/to/image.jpg")

# Identify cat
cat_id, confidence = app.identify_cat("path/to/unknown.jpg")
```

## Monitoring

- Logs: `production/logs/app.log`
- Database: `production/database/cat_database.pkl`
- Configuration: `production/config/config.json`
