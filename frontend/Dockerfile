FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Add environment variables for development
ENV NODE_ENV=development
ENV WDS_SOCKET_PORT=0
ENV WATCHPACK_POLLING=true
ENV NEXT_PUBLIC_API_URL=/api
ENV NEXT_PUBLIC_LLM_API_URL=/llm-api

# Create .next directory and set permissions
RUN mkdir -p .next && chmod -R 777 .next

EXPOSE 3000

# Start the development server
CMD ["npm", "run", "dev"] 