# LLM-Powered-Dashboard-SQL

A full-stack application that combines SQL data analysis with LLM-powered insights to create an intelligent analytics dashboard.

## Architecture

This project uses a microservices architecture with the following components:

- **Frontend**: Next.js 15 application with React, TypeScript, and Tailwind CSS
- **Backend API**: Flask REST API for data management and processing
- **LLM Backend**: Specialized Flask service for LLM-powered analytics
- **Database**: PostgreSQL for data storage
- **Vector Database**: ChromaDB for storing and querying embeddings
- **Nginx**: Web server for routing and load balancing

The components communicate via REST APIs and are containerized using Docker.

## Features

- Data visualization and analytics dashboards
- Natural language query processing for data analytics
- SQL data management and exploration
- LLM-powered insights and report generation
- Vector search capabilities for similar queries and examples

## Prerequisites

- Docker and Docker Compose
- GROQ API key (for LLM integration)

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LLM-Powered-Dashboard-SQL.git
   cd LLM-Powered-Dashboard-SQL
   ```

2. Create a `.env` file in the root directory with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. Start the application using Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Access the application:
   - Frontend: http://localhost
   - Backend API: http://localhost/api
   - LLM API: http://localhost/llm-api

## Component Details

### Frontend

A modern web application built with Next.js, React, and TypeScript. Features include:
- Interactive dashboards and data visualizations
- Form components with validation
- Dark/light mode support
- Responsive design
- API integration with backend services

### Backend API

Flask-based REST API that provides:
- Data management endpoints
- User authentication and authorization
- SQL database access and query execution
- Business logic implementation

### LLM Backend

Specialized service that provides:
- Natural language query processing
- Integration with LLM providers (via GROQ)
- Vector database management for embeddings
- Report generation and data analysis

### Data Persistence

- **PostgreSQL**: Relational database for structured data storage
- **ChromaDB**: Vector database for storing embeddings and semantic search

## Development

Each component can be developed independently:

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### LLM Backend
```bash
cd llm-backend
pip install -r requirements.txt
python app.py
```

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.