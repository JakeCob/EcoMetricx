# 🚄 Railway Deployment Checklist

## Pre-Deployment Checklist

### ✅ **Repository Ready**
- [ ] All code committed to GitHub
- [ ] .env files are NOT committed (only .env.example files)  
- [ ] .gitignore properly configured
- [ ] All Dockerfiles and railway.json files present

### ✅ **External Services Ready**
- [ ] PostgreSQL database accessible (Railway/external)
- [ ] Qdrant vector database accessible (Railway/external)
- [ ] Database contains document data and embeddings

## 🚀 **Railway Deployment Steps**

### 1. **Create Railway Project**
```bash
# Login to Railway CLI (optional)
railway login

# Or use Railway Dashboard
```

### 2. **Deploy API Service (Backend)**
1. **Add Service** → Connect GitHub repository
2. **Settings** → Set root directory: `services/retrieval_api`
3. **Variables** → Add these environment variables:
   ```
   DATABASE_URL=postgresql://user:pass@host:port/db
   API_KEY=your-secure-api-key-here
   QDRANT_URL=https://your-qdrant-url
   QDRANT_API_KEY=your-qdrant-key
   QDRANT_COLLECTION=ecometricx
   FUSION_ALPHA=0.6
   ENABLE_RERANKER=false
   ```
4. **Deploy** → Wait for build to complete
5. **Copy the API URL** (e.g., `https://service-name.up.railway.app`)

### 3. **Deploy Frontend Service**
1. **Add Service** → Connect same GitHub repository  
2. **Settings** → Set root directory: `ui/frontend`
3. **Variables** → Add these environment variables:
   ```
   REACT_APP_API_URL=https://your-api-service-url
   REACT_APP_API_KEY=your-api-key-here
   ```
4. **Deploy** → Wait for build to complete

## 🔍 **Verify Deployment**

### API Service Health Checks
- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] `GET /debug/config` shows correct configuration  
- [ ] Search endpoint works: `POST /search` with API key

### Frontend Service  
- [ ] Homepage loads without errors
- [ ] Search interface appears
- [ ] API calls work (check Network tab)
- [ ] Search returns results

## 🛠️ **Troubleshooting**

### API Issues
| Problem | Solution |
|---------|----------|
| "DATABASE_URL not set" | Add DATABASE_URL to Railway environment variables |
| "Forbidden" errors | Verify API_KEY matches between services |
| "CORS errors" | Check REACT_APP_API_URL is correct |

### Frontend Issues  
| Problem | Solution |
|---------|----------|
| "Network Error" | Verify REACT_APP_API_URL points to API service |
| Blank page | Check browser console for errors |
| "No results" | Verify API is responding and has data |

## 📊 **Success Criteria**
- ✅ Both services deploy without errors
- ✅ Health endpoints respond correctly
- ✅ Frontend can search and display results
- ✅ No security warnings or exposed credentials

## 🔐 **Security Notes**
- Never commit .env files with real credentials
- Use Railway's environment variable dashboard
- API keys should be complex and secure
- Database should use SSL connections in production

## 📞 **Support**
If deployment fails:
1. Check Railway build logs
2. Verify all environment variables are set
3. Test database connectivity
4. Check API endpoint manually with curl/Postman