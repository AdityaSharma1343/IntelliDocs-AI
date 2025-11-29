# 🧠 IntelliDocs AI - Intelligent Document Search Portal

![Version](https://img.shields.io/badge/version-2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Azure](https://img.shields.io/badge/azure-deployed-success.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **Smart Document Management & AI-Powered Search System** built with Azure AI Search, FastAPI, and modern web technologies.

## 🌟 Features

### 📁 Document Management
- ✅ **Multi-format Support**: PDF, Word (DOCX), Excel (XLSX), PowerPoint (PPTX), CSV, and Text files
- ✅ **Drag & Drop Upload**: Simple and intuitive file upload interface
- ✅ **Organized Folders**: Automatic categorization by document type
- ✅ **Full-Text Preview**: View complete document content with proper formatting
- ✅ **Smart Statistics**: Real-time document count and folder organization

### 🔍 AI-Powered Search
- ✅ **Azure AI Search Integration**: Lightning-fast, intelligent search powered by Azure Cognitive Search
- ✅ **Relevance Ranking**: BM25 algorithm with machine learning-based ranking
- ✅ **Advanced Filters**: Filter by document type, date, and relevance
- ✅ **Search Suggestions**: Auto-complete and search recommendations
- ✅ **Semantic Search**: Understanding context and intent, not just keywords

### 🎨 Modern UI/UX
- ✅ **Dark Mode Support**: Eye-friendly interface for any lighting condition
- ✅ **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- ✅ **Real-time Updates**: Live statistics and instant search results
- ✅ **Beautiful Animations**: Smooth transitions and micro-interactions

### 🤖 Machine Learning
- ✅ **Learning to Rank**: Random Forest-based document ranking
- ✅ **Performance Metrics**: Precision, Recall, F1-Score, and MAP tracking
- ✅ **Feature Engineering**: 10+ query-document relevance features
- ✅ **Continuous Improvement**: Model retraining with user feedback

## 🏗️ Architecture

```
IntelliDocs AI
├── Frontend (Static Web App)
│   ├── HTML5 + CSS3 + Vanilla JavaScript
│   ├── Font Awesome Icons
│   └── Deployed on Azure Static Web Apps
│
├── Backend (FastAPI)
│   ├── Document Processing (PyPDF2, python-docx, openpyxl)
│   ├── Azure AI Search Integration
│   ├── RESTful API Endpoints
│   └── Deployed on Azure App Service
│
├── Search Engine
│   ├── Azure Cognitive Search (Primary)
│   ├── BM25 Ranking Algorithm
│   ├── ML-based Relevance Scoring
│   └── Local JSON Backup (Fallback)
│
└── ML Pipeline
    ├── Feature Extraction
    ├── Random Forest Ranker
    ├── Performance Evaluation
    └── Metrics Dashboard
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Azure Account** (Free tier works!)
- **Git**
- **Azure CLI** (for deployment)

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/intellidocs-ai.git
cd intellidocs-ai
```

### 2. Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from .env.example)
cp .env.example .env

# Edit .env with your Azure credentials
```

### 3. Configure Azure Services

```powershell
# Run Azure setup script
.\azure-setup.ps1

# Or manually create resources:
az login
az group create --name rg-ai-search-project --location eastus
az search service create --name your-search-service --resource-group rg-ai-search-project --sku free
```

### 4. Run Locally

```bash
# Start backend server
cd backend
python main.py

# Backend runs on: http://localhost:8000
# API docs: http://localhost:8000/docs
```

```bash
# Start frontend (in another terminal)
cd frontend
# Open index.html in browser
# Or use Python HTTP server:
python -m http.server 8080
# Frontend runs on: http://localhost:8080
```

## ☁️ Deployment

### Backend Deployment (Azure App Service)

#### Option 1: GitHub Actions (Recommended)

Already configured! Just push to `main` branch:

```bash
git add .
git commit -m "Update backend"
git push origin main
```

GitHub Actions will automatically deploy to Azure.

#### Option 2: Manual ZIP Deploy

```powershell
cd backend
Compress-Archive -Path * -DestinationPath app.zip -Force
az webapp deployment source config-zip \
    --name "your-app-name" \
    --resource-group "rg-ai-search-project" \
    --src "app.zip"
```

### Frontend Deployment (Azure Static Web Apps)

```bash
# Already configured via azure-static-web-apps.yml
# Automatically deploys on push to main branch
```

Or manually:
```bash
az staticwebapp create \
    --name "intellidocs-frontend" \
    --resource-group "rg-ai-search-project" \
    --source "frontend" \
    --location "eastus"
```

## 📚 API Documentation

### Base URL
```
Production: https://intellidocs-aditya-backend.azurewebsites.net
Local: http://localhost:8000
```

### Key Endpoints

#### 1. Upload Document
```http
POST /api/documents/upload-file
Content-Type: multipart/form-data

Response:
{
  "message": "Document indexed successfully",
  "document_id": "uuid",
  "filename": "example.pdf",
  "category": "PDF"
}
```

#### 2. Search Documents
```http
POST /api/search
Content-Type: application/json

{
  "query": "employee handbook",
  "top": 10,
  "filter": "category eq 'PDF'"
}

Response:
{
  "count": 5,
  "results": [
    {
      "id": "uuid",
      "title": "Employee Handbook",
      "content": "...",
      "score": 0.95
    }
  ]
}
```

#### 3. Get Document by ID
```http
GET /api/documents/{document_id}

Response:
{
  "id": "uuid",
  "title": "Document Title",
  "content": "Full document text...",
  "category": "PDF",
  "created_date": "2025-01-10T12:00:00Z"
}
```

#### 4. Delete Document
```http
DELETE /api/documents/{document_id}

Response:
{
  "message": "Document deleted successfully",
  "document_id": "uuid"
}
```

#### 5. Get Statistics
```http
GET /api/stats

Response:
{
  "total_documents": 18,
  "categories": {
    "PDF": 5,
    "Word": 8,
    "Excel": 3
  }
}
```

### Full API Documentation
Visit: `https://your-backend.azurewebsites.net/docs`

## 🤖 Machine Learning Pipeline

### Training the Model

```bash
cd backend
python ml_model.py
```

This will:
1. Load documents from database
2. Train BM25 ranker
3. Train Random Forest ranking model
4. Generate evaluation metrics
5. Create performance visualizations

### Generating Metrics

```bash
python search_metrics.py
```

Outputs:
- `ml_metrics/feature_importance.png`
- `ml_metrics/roc_curve.png`
- `ml_metrics/confusion_matrix.png`
- `ml_metrics/search_quality_report.html`

### Model Performance

| Metric | Score |
|--------|-------|
| Precision | 0.82 |
| Recall | 0.75 |
| F1-Score | 0.78 |
| Accuracy | 0.86 |
| MAP | 0.79 |

## 🛠️ Tech Stack

### Frontend
- HTML5, CSS3, JavaScript (ES6+)
- Font Awesome 6.4.0
- Google Fonts (Poppins)
- Responsive Grid Layout

### Backend
- **Framework**: FastAPI 0.109.0
- **Search**: Azure Cognitive Search SDK 11.4.0
- **Document Processing**:
  - PyPDF2 (PDF)
  - python-docx (Word)
  - openpyxl (Excel)
  - python-pptx (PowerPoint)
- **ML Libraries**:
  - scikit-learn 1.3.2
  - numpy 1.24.3
  - pandas 2.1.4
- **Visualization**:
  - matplotlib 3.7.2
  - seaborn 0.12.2

### Infrastructure
- **Hosting**: Azure App Service (Backend) + Azure Static Web Apps (Frontend)
- **Search**: Azure Cognitive Search (Free Tier)
- **CI/CD**: GitHub Actions
- **Storage**: Azure Search Index + Local JSON Backup

## 📊 Project Structure

```
intellidocs-ai/
├── .github/
│   └── workflows/
│       ├── azure-backend-deploy.yml
│       └── azure-static-web-apps.yml
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── document_processor.py   # Document text extraction
│   ├── ml_model.py            # Machine learning pipeline
│   ├── search_metrics.py      # Performance evaluation
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables
├── frontend/
│   └── index.html             # Single-page application
├── ml_metrics/                # Generated ML reports
├── ml_models/                 # Trained model files
├── uploaded_files/            # Temporary upload directory
├── azure-setup.ps1            # Azure resource setup script
├── README.md
└── .gitignore
```

## 🔐 Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Azure Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=rg-ai-search-project
AZURE_SEARCH_SERVICE_NAME=your-search-service
AZURE_SEARCH_ADMIN_KEY=your-admin-key
AZURE_SEARCH_ENDPOINT=https://your-service.search.windows.net
AZURE_SEARCH_INDEX_NAME=employee-documents
AZURE_SEARCH_QUERY_KEY=your-query-key

# Application Settings
USE_AZURE_PRIMARY=true
FALLBACK_TO_LOCAL=true
APP_ENV=production
APP_PORT=8000
APP_HOST=0.0.0.0
```

## 🧪 Testing

```bash
# Test backend endpoints
curl http://localhost:8000/

# Test search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top": 5}'

# Test document upload
curl -X POST http://localhost:8000/api/documents/upload-file \
  -F "file=@test.pdf"
```

## 🐛 Troubleshooting

### Issue: Excel files not processing
**Solution**: Ensure openpyxl is installed
```bash
pip install openpyxl
```

### Issue: Azure Search 405 error
**Solution**: Redeploy backend
```bash
cd backend
az webapp restart --name your-app-name --resource-group rg-ai-search-project
```

### Issue: Documents not displaying full content
**Solution**: Check API endpoint returns full content
```bash
curl https://your-backend.azurewebsites.net/api/documents/{doc-id}
```

## 📈 Roadmap

- [ ] Multi-language support
- [ ] PDF annotation and highlighting
- [ ] Document version control
- [ ] Collaborative document editing
- [ ] Advanced analytics dashboard
- [ ] Mobile app (React Native)
- [ ] OCR for scanned documents
- [ ] Integration with Microsoft 365

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Aditya Sharma**
- GitHub: [@AdityaSharma](https://github.com/AdityaSharma1343)
- Email: aditya134766@gmail.com
- LinkedIn: [Aditya Sharma](https://linkedin.com/in/aditya-sharma-225488224)

## 🙏 Acknowledgments

- Azure Cognitive Search for powerful search capabilities
- FastAPI for the amazing web framework
- Font Awesome for beautiful icons
- OpenAI for inspiration

## 📞 Support

For issues and questions:
- 📧 Email: aditya134766@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/AdityaSharma1343/intellidocs-ai/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/AdityaSharma1343/intellidocs-ai/discussions)

---

**Made with ❤️ by Aditya Sharma**

⭐ Star this repo if you found it helpful!