Food-Ops-Copilot - Sales Analytics

A GenAI-powered SQL copilot application for restaurant sales data analysis using BigQuery and Chainlit. This deployment is specifically optimized for sales transaction analysis with comprehensive payment, tax, and geographic insights.

🏗️ Architecture Overview
The application is built around a natural language to SQL conversion pipeline with these key components:

Core Components
app.py
 - Chainlit Chat Interface
Interactive web-based chat UI using Chainlit framework
Handles user messages and displays SQL results
Integrates with BigQuery for data export
Uses toggle buttons for SQL/metadata visibility
main.py
 - Entry Point & Processing Engine
Supports 3 modes: single question, batch processing, replay
Handles multiprocessing for batch operations
Orchestrates the entire question-to-answer pipeline
sql_agent.py
 - Core NL-to-SQL Engine
Uses Vertex AI Gemini models for SQL generation
Implements few-shot learning with examples
Handles schema understanding and query validation
Integrates with reflection and embedding components
Supporting Modules
reflection_agent.py
 - Query validation and improvement
embedding_model.py
 - Semantic similarity for few-shot examples
chat_history_manager.py
 - Conversation context management
config.py
 - GCP and application configuration
prompt.py
 - Prompt templates for the AI models
🎯 Key Features
Natural Language Queries: Ask questions in plain English
SQL Generation: Automatic conversion to BigQuery SQL
Interactive Chat: Web-based conversational interface
Batch Processing: Handle multiple questions simultaneously
Result Visualization: Format and display query results
Export Capabilities: Save results to BigQuery tables
🔧 Technical Stack
AI Models: Gemini-2.5-pro-preview, Gemini-2.5-flash
Data Platform: Google BigQuery
UI Framework: Chainlit
Cloud Platform: Google Cloud Platform (Vertex AI)
Target Dataset: Zomato Restaurant Dataset (configurable via unified_config.yaml)
📊 Data Flow
User asks a natural language question
SQL Agent processes the question using Gemini models
Few-shot examples are retrieved using embeddings
SQL query is generated and validated
Query executes against BigQuery
Results are formatted and displayed to user
Conversation history is maintained for context
This appears to be a business intelligence copilot specifically designed for food operations data analysis, allowing non-technical users to query complex datasets using natural language.# Auto-deployment test - Tue Jul 22 04:38:55 IST 2025
