# EcoMetricx Railway Deployment Guide

## Overview
This app consists of two services that need to be deployed to Railway:
1. **API Service** (Python/FastAPI) - Backend search API
2. **Frontend Service** (React) - User interface

## Pre-Deployment Setup

### 1. Database & Vector Store
Ensure you have:
- ✅ PostgreSQL database with search data
- ✅ Qdrant vector database with embeddings  
- ✅ Database connection strings ready

### 2. Push to GitHub
```bash
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

## Railway Deployment Steps

### 1. Deploy API Service First

1. **Create New Project** in Railway dashboard
2. **Connect GitHub** repository 
3. **Add Service** → Select the repository
4. **Set Root Directory**: `services/retrieval_api`
5. **Configure Environment Variables**:
   ```
   DATABASE_URL=postgresql://postgres:password@host:port/database
   API_KEY=your-api-key-here
   QDRANT_URL=https://your-qdrant-url
   QDRANT_API_KEY=your-qdrant-api-key
   QDRANT_COLLECTION=ecometricx
   FUSION_ALPHA=0.6
   ENABLE_RERANKER=false
   PORT=8000
   ```

6. **Deploy** - Railway will use the `Dockerfile` automatically
7. **Copy the API Service URL** (e.g., `https://api-service-production-abcd.up.railway.app`)

### 2. Deploy Frontend Service

1. **Add Service** to the same Railway project
2. **Connect the same GitHub** repository
3. **Set Root Directory**: `ui/frontend` 
4. **Configure Environment Variables**:
   ```
   REACT_APP_API_URL=https://your-api-service-url
   REACT_APP_API_KEY=your-api-key-here
   API_SERVICE_URL=https://your-api-service-url
   API_KEY=your-api-key-here
   ```

5. **Deploy** - Railway will use the `Dockerfile` automatically

## Environment Variables Reference

### API Service Required Variables
- `DATABASE_URL` - PostgreSQL connection string
- `API_KEY` - Authentication key for API access
- `QDRANT_URL` - Qdrant vector database URL  
- `QDRANT_API_KEY` - Qdrant authentication key
- `QDRANT_COLLECTION` - Collection name (usually "ecometricx")

### Frontend Required Variables  
- `REACT_APP_API_URL` - URL of the deployed API service
- `REACT_APP_API_KEY` - API authentication key

## Health Checks
- **API**: `GET /health` - Returns `{"status": "ok"}`
- **Frontend**: `GET /` - Returns the React app

## Troubleshooting

### API Service Issues
1. Check logs for database connection errors
2. Verify all environment variables are set
3. Test `/health` endpoint
4. Test `/debug/config` for configuration validation

### Frontend Issues  
1. Check if API URL is correct in environment variables
2. Verify CORS is working (API should return proper headers)
3. Check browser network tab for API call errors

### Database Connection
Ensure your DATABASE_URL includes:
- Correct host, port, username, password
- SSL mode if required (`?sslmode=require`)
- Database name

### CORS Issues
The API includes CORS middleware allowing all origins. If issues persist:
- Check API logs for CORS-related errors  
- Verify API URL is correct in frontend environment variables

## Success Criteria
✅ API service responds to health checks  
✅ Frontend loads and displays search interface
✅ Search queries return results from the database
✅ No CORS errors in browser console

## Post-Deployment
After successful deployment:
1. Test search functionality with various queries
2. Monitor logs for errors
3. Set up custom domains if needed
4. Configure monitoring/alerts

## Support
If deployment fails, check:
1. Railway build logs
2. Application logs  
3. Environment variable configuration
4. Database connectivity