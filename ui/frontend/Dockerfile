# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Remove any existing .env files to prevent circular references
RUN rm -f .env* || true

# Create a clean .env file for build
RUN echo "REACT_APP_API_URL=https://ecometricx-production.up.railway.app" > .env && \
    echo "REACT_APP_API_KEY=placeholder" >> .env

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app to nginx
COPY --from=build /app/build /usr/share/nginx/html

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]