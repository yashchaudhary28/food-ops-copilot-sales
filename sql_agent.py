# sql_agent.py

import json
import re
import ast
import numpy as np
import pandas
from typing import Optional
import pandas as pd
from google.cloud import bigquery
from difflib import get_close_matches
from vertexai import init
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.generative_models import GenerativeModel
from prompt import prompt_render, suggestive_prompt_render, metadataExtractor_prompt, conversation_prompt_render
from logger_util import setup_logger
from logger_context import call_id_var
from chat_history_manager import ChatHistoryManager
from config_loader import Config

# Initialize Vertex AI with config settings
init(project=Config.gcp.project_id, location=Config.gcp.vertex_ai_location)
logger = setup_logger(__name__)

class SqlAgent:
    def __init__(
        self,
        question: str,
        table_name: str,
        schema_path: str,
        fewshotInfo_path: str,
        model_name: str,
        reflection_model_name: str,
        embedding_store: pd.DataFrame,
        project_id: Optional[str] = None,
        reflection_agent=None,
        embedding_model = None,
        chat_history_manager = None,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 1
    ):
        self.question = question
        self.table_name = table_name
        self.schema_path = schema_path
        self.fewshotInfo_path = fewshotInfo_path
        self.model_name = model_name
        self.reflection_model_name = reflection_model_name
        self.embedding_store = embedding_store
        # Use config project_id if not provided
        self.project_id = project_id or Config.gcp.project_id
        self.reflection_agent = reflection_agent
        self.embedding_model = embedding_model
        self.chat_history_manager = chat_history_manager or ChatHistoryManager()
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        self.schema = None
        self.fewshot_info = None
        self.info = None
        self.fewshot_sql = None
        self.response = None
        self.query = None
        self.suggestions = None
        self.conversation_response = None
        self.chat_history = None
        
    def __repr__(self):
        return f"<SqlAgent(question='{self.question[:30]}...', table='{self.table_name}')>"

    def read_schema(self) -> None:
        logger.info(f"Reading schema from: {self.schema_path}")
        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
        except FileNotFoundError:
            logger.critical(f"Schema file not found at: {self.schema_path}")
            raise
        except json.JSONDecodeError as e:
            logger.critical(f"Malformed JSON in schema file: {e}")
            raise
        except Exception as e:
            logger.error("Unexpected error reading schema", exc_info=True)
            raise

        header = "Column Name | Data Type | Description | Mode | Sample\n"
        divider = "-" * 80 + "\n"
        schema_str = header + divider

        # Handle both array and dictionary schema formats
        if isinstance(schema_data, list):
            # New array format from schema_config.json
            for column in schema_data:
                row = [
                    str(column.get("name", "N/A")),
                    str(column.get("type", "N/A")),
                    str(column.get("description", "N/A")),
                    str(column.get("mode", "N/A")),
                    str(column.get("sample", "N/A"))
                ]
                schema_str += " | ".join(row) + "\n"
        elif isinstance(schema_data, dict):
            # Legacy dictionary format (backward compatibility)
            for col, details in schema_data.items():
                row = [
                    str(col),
                    str(details.get("Data_type", "N/A")),
                    str(details.get("description", "N/A")),
                    str(details.get("is_nullable", "N/A")),
                    str(details.get("sample", "N/A"))
                ]
                schema_str += " | ".join(row) + "\n"
        else:
            logger.error(f"Unexpected schema format: {type(schema_data)}")
            raise ValueError(f"Schema must be a list or dictionary, got {type(schema_data)}")

        self.schema = schema_str
        logger.debug("Schema formatted successfully.")
    
    def read_few_shot_info(self) -> None:
        logger.info(f"Reading few-shot info from: {self.fewshotInfo_path}")
        try:
            with open(self.fewshotInfo_path, "r", encoding="utf-8") as f:
                self.fewshot_info = f.read()
            logger.debug("Few-shot info loaded successfully.")
        except FileNotFoundError:
            logger.critical(f"Few-shot info file not found: {self.fewshotInfo_path}")
            raise
        except Exception as e:
            logger.error("Error reading few-shot info examples", exc_info=True)
            raise
    
    
    def metadata_extractor(self) -> None:
        logger.info("Generating bins, column, intent, date interval information from model...")
        try:
            self.chat_history = self.chat_history_manager.get_last_n()
            logger.info(f"Chat History Recieved: {self.chat_history}")
            prompt = metadataExtractor_prompt(self.question, self.schema, self.fewshot_info, self.chat_history)
            logger.debug(f"Generated prompt:\n{prompt}")

            model = GenerativeModel(self.model_name)
            config = {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k}
            result = model.generate_content(prompt, generation_config=config)
            self.info = result.text
            self.clean_json_string()
            if "data_related" not in self.info:
                self.info["data_related"] = 1
            try:
                with open(self.schema_path, "r", encoding="utf-8") as f:
                    raw_schema = json.load(f)
                    
                # Handle both array and dictionary formats
                if isinstance(raw_schema, list):
                    # Convert array format to dictionary format for compatibility
                    schema_data = {item['name']: item for item in raw_schema}
                    logger.info(f"✅ Schema converted from array to dictionary format ({len(schema_data)} columns)")
                elif isinstance(raw_schema, dict):
                    # Already in dictionary format
                    schema_data = raw_schema
                    logger.info(f"✅ Schema loaded in dictionary format ({len(schema_data)} columns)")
                else:
                    raise ValueError(f"Unexpected schema format: {type(raw_schema)}")
                    
            except FileNotFoundError:
                logger.critical(f"Schema file not found at: {self.schema_path}")
                raise
            except json.JSONDecodeError as e:
                logger.critical(f"Malformed JSON in schema file: {e}")
                raise
            except Exception as e:
                logger.error("Unexpected error reading schema", exc_info=True)
                raise
            self.info["relevant_columns_info"] = {}
            updated_columns = []
            for col in self.info.get("relevant_columns",[]):
                if col in schema_data:
                    self.info["relevant_columns_info"][col] = schema_data[col]
                    updated_columns.append(col)
                else:
                    possible_matches = get_close_matches(col, schema_data.keys(), n=1, cutoff=0.6)
                    if possible_matches:
                        best_match = possible_matches[0]
                        self.info["relevant_columns_info"][best_match] = schema_data[best_match]
                        logger.warning(f"Column '{col}' not found in schema. Replaced with '{best_match}'.")
                        updated_columns.append(best_match)
                    else:
                        logger.warning(f"Column '{col}' not found in schema and no close match was found.")
            # Update list
            self.info["relevant_columns"] = updated_columns
            self.info["bins_info"] = None
            if self.info.get("binning_required") == 1:
                if self.info.get("dpd_question") == 1:
                    self.info["bins_info"] = [0, 1, 2, 3, 4]
                else:
                    cols = self.info.get("bins", [])
                    for col in cols:
                        sample = schema_data.get(col, {}).get("sample", [])
                        if not sample or len(sample) < 2:
                            logger.warning(f"Skipping binning for column '{col}': insufficient sample data.")
                            continue
                        min_val, max_val = min(sample), max(sample)
                        bins = [int(min_val + (max_val - min_val) * q) for q in [0, 0.25, 0.5, 0.75, 1.0]]
                        self.info["bins_info"] = bins
            logger.info("Meta Data Extracted")

        except Exception as e:
            logger.critical("Model generation failed", exc_info=True)
            raise
    
    def clean_json_string(self)->None:
        self.info = re.sub(r'^```json\s*|\s*```$', '', self.info.strip(), flags=re.IGNORECASE)
        self.info = json.loads(self.info)
    
    def read_few_shot_examples(self) -> None:
        try:
            logger.info("Reading few shot example....")
            
            # TEMPORARY: Skip embedding logic and use basic few-shot examples
            logger.info("⚠️ Bypassing embedding logic - using basic few-shot examples")
            self.fewshot_info = """Example 1:
Question: What are the top restaurants by rating?
SQL: SELECT restaurant_name, rating FROM restaurants ORDER BY rating DESC LIMIT 10;

Example 2:
Question: Show restaurants in a specific city
SQL: SELECT * FROM restaurants WHERE city = 'Bangalore';

Example 3:
Question: What's the average cost for different cuisines?
SQL: SELECT cuisine, AVG(cost_for_two) as avg_cost FROM restaurants GROUP BY cuisine;"""
            return
            
            # Original embedding logic (temporarily disabled)
            if self.embedding_model is None:
                raise ValueError("Embedding model is not initialized.")
            if self.embedding_store is None or "embedding" not in self.embedding_store.columns:
                raise ValueError("Embedding store is missing or malformed.")
            if not self.question or not isinstance(self.question, str):
                raise ValueError("A valid input question is required.")
            embedding_model = self.embedding_model.get_model()
            embedding_response = embedding_model.get_embeddings([self.question])
            if not embedding_response or not hasattr(embedding_response[0], "values"):
                raise ValueError("Failed to generate embedding for the input question.")
            input_embedding = np.array(embedding_response[0].values).reshape(1, -1)
            self.embedding_model.parse_embeddings_column(self.embedding_store)
            embeddings_array = np.vstack(self.embedding_store["embedding"].to_numpy())
            similarities = cosine_similarity(input_embedding, embeddings_array)[0]
            scored_data = self.embedding_store.copy()
            scored_data["similarity"] = similarities
            top_examples = scored_data.sort_values(by="similarity", ascending=False).head(3)
            prompt_lines = []
            for _, row in top_examples.iterrows():
                question = str(row.get("question", "")).strip()
                query = str(row.get("query", "")).strip()
                if question and query:
                    prompt_lines.append(f"### Question:\n{question}\n### SQL:\n{query}\n")
            self.fewshot_sql = "\n".join(prompt_lines).strip()
            logger.debug(f"Few-shot prompt created successfully: {self.fewshot_sql}")
        except Exception as e:
            logger.critical("Failed to generate few-shot examples.", exc_info=True)
            self.fewshot_sql = ""
            raise

    def generate_response(self) -> Optional[str]:
        logger.info("Generating SQL from model...")
        try:
            self.chat_history = self.chat_history_manager.get_last_n()
            logger.debug(f"Chat History Recieved: {self.chat_history}")
            prompt = prompt_render(self.question, self.info, self.fewshot_sql, self.table_name, self.chat_history)
            logger.debug(f"Generated prompt:\n{prompt}")

            model = GenerativeModel(self.model_name)
            config = {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k}
            result = model.generate_content(prompt, generation_config=config)
            self.response = result.text
            self.query = self.extract_query()
            return self.query
        except Exception as e:
            logger.critical("Model generation failed", exc_info=True)
            raise
    
    def extract_query(self) -> Optional[str]:
        logger.info("Extracting SQL query from model output...")
        try:
            match = re.search(r"```sql\s*((?:SELECT|WITH).*?)\s*```", self.response, re.IGNORECASE | re.DOTALL)
            if not match:
                logger.warning("No valid SQL block found in LLM response.")
                return None
            query = match.group(1).strip()
            if re.match(r"^(INSERT|UPDATE|DELETE|MERGE)", query, re.IGNORECASE):
                logger.error("DML statements are not permitted.")
                raise ValueError("Disallowed DML operation detected.")
            logger.debug(f"Extracted SQL:\n{query}")
            return query
        except Exception as e:
            logger.error("Query extraction failed.", exc_info=True)
            return None

    def regenerate_query_with_reflection(self) -> Optional[str]:
        try:
            logger.info("Regenerating query using reflection suggestions...")
            prompt = suggestive_prompt_render(self.question, self.query, self.suggestions)
            logger.debug("New prompt after reflection:\n" + prompt)

            model = GenerativeModel(self.reflection_model_name)
            config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k
            }

            result = model.generate_content(prompt, generation_config=config)
            self.response = result.text
            self.query = self.extract_query()
            return self.query
        except Exception as e:
            logger.error("Failed to regenerate query with reflection.", exc_info=True)
            return None

    def run_query(self) -> Optional[pd.DataFrame]:
        # Initialize BigQuery client with project from config
        client = bigquery.Client(project=self.project_id)
        logger.info("Starting query execution...")

        max_attempts = 3
        attempt = 0
        current_query = self.query

        while attempt < max_attempts:
            try:
                logger.info(f"Attempt {attempt + 1} - Running query")
                logger.debug(f"Query:\n{current_query}")
                df = client.query(current_query).to_dataframe()
                logger.info(f"Query succeeded on attempt {attempt + 1}. Rows: {len(df)}")
                return df

            except Exception as e:
                logger.error(f"Query failed on attempt {attempt + 1}: {e}", exc_info=True)

                if not self.reflection_agent:
                    logger.warning("Reflection agent not available. Aborting retry.")
                    break

                logger.info("Generating reflection suggestions...")
                self.suggestions = self.reflection_agent.reflect(
                    question=self.question,
                    failed_query=current_query,
                    error_msg=str(e),
                )

                if not self.suggestions:
                    logger.warning("No suggestions returned by reflection agent.")
                    break

                current_query = self.regenerate_query_with_reflection()
                logger.info(f"Query after reflection: {current_query}")
                if not current_query:
                    logger.warning("Failed to regenerate query from suggestions.")
                    break

                attempt += 1

        logger.critical("All query attempts failed after reflection.")
        return None
    
    def conversation_bot(self) -> None:
        logger.info("Handling as a general conversation (not data related)...")
        try:
            self.chat_history = self.chat_history_manager.get_last_n()
            logger.debug(f"Chat History Recieved: {self.chat_history}")
            prompt = conversation_prompt_render(self.question, self.chat_history)
            logger.debug(f"Conversation prompt:\n{prompt}")

            model = GenerativeModel(self.model_name)
            config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k
            }

            result = model.generate_content(prompt, generation_config=config)
            self.conversation_response = result.text.strip()
            logger.info("Conversation response generated.")
    
        except Exception as e:
            logger.error("Conversation bot generation failed.", exc_info=True)
            self.conversation_response = "Sorry, I encountered an issue while trying to respond."

    def run(self) -> Optional[str]:
        logger.info("Running SQL Agent pipeline...")
        self.read_schema()
        self.read_few_shot_info()
        self.metadata_extractor()

        df = None  # initialize df to None

        if self.info["data_related"] == 1:
            self.read_few_shot_examples()
            self.generate_response()

            if self.query:
                df = self.run_query()
        else:
            self.conversation_bot()

        # Save to chat history
        self.chat_history_manager.append(
            question=self.question,
            info=self.info,
            query=self.query,
            conversation_response=self.conversation_response,
        )
        logger.info("Appended the chat history")
        return self.info, self.query, self.conversation_response, df

