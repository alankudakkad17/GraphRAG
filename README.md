# Hybrid GraphRAG API

## Project Description
The Hybrid GraphRAG API provides a robust framework for graph-based reasoning and generation. Built to facilitate powerful data manipulation and retrieval, it harnesses the strengths of both hybrid architectures and graph theory.

## Features
- **Flexible Querying**: Supports complex querying of data graphs.
- **Performance Optimization**: Efficient algorithms designed to process large datasets with minimal latency.
- **Extensible Architecture**: Easily integrate with other systems and extend functionality.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/alankudakkad17/GraphRAG.git
   cd GraphRAG
   ```
2. **Install Dependencies**:
   ```bash
   npm install
   ```
3. **Run the Application**:
   ```bash
   npm start
   ```

## API Endpoints
### 1. GET /api/graphs
- **Description**: Retrieves all graphs from the database.
- **Response**: Returns a list of graph objects.

### 2. POST /api/graphs
- **Description**: Creates a new graph.
- **Request Body**: JSON object representing the graph.
- **Response**: Returns the created graph object.

### 3. DELETE /api/graphs/:id
- **Description**: Deletes a graph by ID.
- **Response**: Confirmation message.

## Requirements
- Node.js (version 14 or higher)
- Express.js
- MongoDB
- Mongoose

Ensure you have all the required dependencies installed before running the API.