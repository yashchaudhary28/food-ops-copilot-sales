"""
Main entry point for the Food-Ops-Bot application.

This module provides functionality to process natural language questions about data
in BigQuery and return SQL queries and results. It supports both single question
processing and batch processing of multiple questions.
"""
import os
import sys
import time
import uuid
import pandas as pd
from multiprocessing import Pool
from typing import Dict, Any, List, Optional, Tuple

from sql_agent import SqlAgent
from reflection_agent import ReflectionAgent
from embedding_model import EmbeddingModel
from chat_history_manager import ChatHistoryManager
from logger_util import setup_logger
from logger_context import call_id_var
from config_loader import Config

# Set up logging
logger = setup_logger("main")

def process_single_question(args: Tuple[str, str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Process a single question using the SQL agent with configuration from config files.
    
    Args:
        args: Tuple containing:
            - user_id: Unique identifier for the user/session
            - question: The natural language question to process
            - config_overrides: Optional dictionary with configuration overrides
            
    Returns:
        Dictionary containing the processing results including:
        - user_id: The user/session ID
        - question: The original question
        - query: The generated SQL query
        - rows: Query results as a list of dictionaries
        - conversation_response: Natural language response
        - time_taken_sec: Processing time in seconds
    """
    user_id, question, config_overrides = args
    call_id_var.set(user_id)
    start_time = time.time()
    
    # Initialize configuration with any overrides
    config = {
        "table_name": config_overrides.get("table_name") if config_overrides else None,
        "model_name": config_overrides.get("model_name") if config_overrides else None,
        "embedding_store_path": config_overrides.get("embedding_store_path") or Config.application.embedding_store_path
    }
    
    # Set default table name if not provided
    if not config["table_name"]:
        config["table_name"] = Config.get_full_table_name()
    
    # Get table-specific configuration
    # Use unified config for table configuration
    project_id = Config.gcp.project_id
    
    try:
        # Initialize components with configuration from config files
        logger.info(f"Initializing components for table: {config['table_name']}")
        
        # Initialize embedding model
        embedding_model = EmbeddingModel()
        
        # Initialize reflection agent
        reflection_agent = ReflectionAgent(
            model_name=Config.models.reflection.name,
            temperature=0.0,
            top_p=0.0,
            top_k=1
        )
        
        # Initialize chat history manager
        chat_history_manager = ChatHistoryManager()
        
        # Load embedding store if it exists
        embedding_store = pd.DataFrame()
        if os.path.exists(config["embedding_store_path"]):
            try:
                embedding_store = pd.read_csv(config["embedding_store_path"])
                logger.info(f"Loaded embedding store from {config['embedding_store_path']}")
            except Exception as e:
                logger.warning(f"Failed to load embedding store: {e}")
        
        # Create SQL agent with configuration
        agent = SqlAgent(
            question=question,
            table_name=config["table_name"],
            schema_path=Config.application.schema_config_path,
            fewshotInfo_path=Config.application.few_shot_examples_path,
            model_name=config["model_name"] or Config.models.primary.name,
            reflection_model_name=Config.models.reflection.name,
            embedding_store=embedding_store,
            project_id=project_id,
            reflection_agent=reflection_agent,
            embedding_model=embedding_model,
            chat_history_manager=chat_history_manager
        )

        info, query, conversation_response, df = agent.run()
        logger.info(f"Info Generated: {info}")
        result_preview = df.to_dict(orient="records") if df is not None else None

    except Exception as e:
        logger.error(f"Error while processing question: {question}", exc_info=True)
        query = None
        result_preview = None
        conversation_response = "Error occurred during processing."

    end_time = time.time()
    return {
        'user_id': user_id,
        'question': question,
        'query': query,
        'rows': result_preview,
        'conversation_response': conversation_response,
        'time_taken_sec': end_time - start_time
    }

def multiprocessor():
    """
    Process multiple questions in parallel using multiprocessing.
    
    This function reads questions from a CSV file and processes them in parallel
    using multiple worker processes. Results are saved to a timestamped CSV file.
    """
    try:
        # Get default configuration
        # Using unified config system - no need to check BIGQUERY_TABLES
        default_table = Config.get_full_table_name()
        
        # Get user input
        csv_file = input("Enter the path to the CSV file with a 'question' column: ")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
        # Read and validate input data
        data = pd.read_csv(csv_file)
        if 'question' not in data.columns:
            raise ValueError("CSV must contain a 'question' column.")
            
        # Load embedding store if it exists
        embedding_store_path = Config.application.embedding_store_path
        config_overrides = {
            "table_name": default_table,
            "embedding_store_path": embedding_store_path,
            "model_name": Config.models.primary.name
        }
        
        # Get batch size from user input (default to all rows)
        batch_size = input(f"Enter batch size (default: all {len(data)} rows): ").strip()
        if batch_size and batch_size.isdigit():
            batch_size = int(batch_size)
            data_to_process = data.head(batch_size)
            logger.info(f"Processing {batch_size} questions from CSV")
        else:
            data_to_process = data
            logger.info(f"Processing all {len(data)} questions from CSV")
        
        # Prepare tasks for processing
        tasks = [
            (str(uuid.uuid4()), row['question'], config_overrides)
            for _, row in data_to_process.iterrows()
        ]

        # Get worker count from user input (default to optimal count)
        optimal_workers = min(10, len(tasks), os.cpu_count() or 4)
        worker_input = input(f"Enter number of workers (default: {optimal_workers}): ").strip()
        if worker_input and worker_input.isdigit():
            worker_count = int(worker_input)
        else:
            worker_count = optimal_workers
        
        # Process questions in parallel
        logger.info(f"Starting multiprocessing with {worker_count} workers for {len(tasks)} tasks")
        try:
            with Pool(worker_count) as pool:
                results = pool.map(process_single_question, tasks)
        except Exception as e:
            logger.error(f"Error during parallel processing: {e}")
            raise

        # Save results to a CSV file
        output_file = f"output_{int(time.time())}.csv"
        pd.DataFrame(results).to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error in multiprocessor: {e}", exc_info=True)
        raise

def main():
    # Use unified configuration system - NO hardcoded values
    table_name = f"{Config.bigquery.target_table.project_id}.{Config.bigquery.target_table.dataset_id}.{Config.bigquery.target_table.table_id}"
    schema_path = Config.application.schema_config_path
    fewshotInfo_path = Config.application.few_shot_examples_path
    model_name = Config.models.primary.name
    reflection_model_name = Config.models.reflection.name
    # Get project ID from config
    project_id = Config.gcp.project_id
    
    # Load or create embedding store
    try:
        embedding_store = pd.read_csv("embedding_store.csv")
        logger.info("Loaded existing embedding store")
    except FileNotFoundError:
        # Create empty embedding store if file doesn't exist
        embedding_store = pd.DataFrame(columns=['question', 'embedding'])
        logger.info("Created empty embedding store (file not found)")
    except Exception as e:
        logger.error(f"Error loading embedding store: {e}")
        embedding_store = pd.DataFrame(columns=['question', 'embedding'])

    logger.info("========== SQL Agent Conversational Mode Started ==========")

    try:
        user_id = str(uuid.uuid4())
        call_id_var.set(user_id)

        embedding_model = EmbeddingModel(project_id=project_id)
        reflection_agent = ReflectionAgent(
            model_name=reflection_model_name,
            temperature=0.0,
            top_p=0.0,
            top_k=1
        )
        chat_history_manager = ChatHistoryManager()

        while True:
            question = input("\nAsk your question (or type 'exit' to quit):\n").strip()
            if question.lower() in {"exit", "quit"}:
                print("Exiting conversation. Goodbye!")
                break

            call_id_var.set(user_id)

            agent = SqlAgent(
                question=question,
                table_name=table_name,
                schema_path=schema_path,
                fewshotInfo_path=fewshotInfo_path,
                model_name=model_name,
                reflection_model_name=reflection_model_name,
                embedding_store=embedding_store,
                project_id=project_id,
                reflection_agent=reflection_agent,
                embedding_model=embedding_model,
                chat_history_manager=chat_history_manager
            )

            info, query, conversation_response, df = agent.run()

            logger.info(f"Info Generated: {info}")

            if query:
                print("\nGenerated SQL Query:\n")
                print(query)

                if df is not None:
                    print("\nQuery Result Preview:\n")
                    print(df)
                else:
                    print("⚠️ Query execution failed, even after reflection.")
            elif conversation_response:
                print("\nAssistant Response:\n")
                print(conversation_response)
            else:
                print("⚠️ No response generated.")

    except Exception as e:
        logger.critical(f"Pipeline failed due to unexpected error: {e}", exc_info=True)
        print("An unexpected error occurred. Check logs for more details.")

    finally:
        logger.info("========== SQL Agent Conversational Mode Finished ==========")

def record_conversation_with_responses(csv_path: str, output_path: str = "conversation_with_responses.csv"):
    """
    Process a CSV file containing questions and save the responses to another CSV file.
    
    Args:
        csv_path: Path to the input CSV file containing questions
        output_path: Path where the output CSV with responses will be saved
        
    Raises:
        FileNotFoundError: If the input CSV file doesn't exist
        ValueError: If the input CSV doesn't contain a 'question' column
    """
    try:
        # Validate input file
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input CSV file not found: {csv_path}")
            
        # Get default configuration from unified config
        embedding_store_path = Config.application.embedding_store_path
        config_overrides = {
            "table_name": Config.get_full_table_name(),
            "embedding_store_path": embedding_store_path
        }
        
        # Read input data
        data = pd.read_csv(csv_path)
        if 'question' not in data.columns:
            raise ValueError("Input CSV must contain a 'question' column.")
            
        # Process each question
        results = []
        for _, row in data.iterrows():
            question = row['question']
            start_time = time.time()
            
            # Create SqlAgent for this question
            agent = SqlAgent(
                question=question,
                table_name=Config.get_full_table_name(),
                schema_path=Config.application.schema_config_path,
                fewshotInfo_path=Config.application.few_shot_examples_path,
                model_name=Config.models.primary.name,
                reflection_model_name=Config.models.reflection.name,
                embedding_store=embedding_store,
                project_id=Config.gcp.project_id,
                reflection_agent=reflection_agent,
                embedding_model=embedding_model,
                chat_history_manager=chat_history_manager
            )

            try:
                info, query, conversation_response, df_result = agent.run()

                query_result = df_result.head(5).to_dict(orient="records") if df_result is not None else None

                results.append({
                    'question': question,
                    'query': query,
                    'response': conversation_response,
                    'query_result_preview': query_result,
                    'total_time': time.time() - start_time
                })

            except Exception as e:
                logger.error(f"Failed to process question: {question}", exc_info=True)
                results.append({
                    'question': question,
                    'query': None,
                    'response': "Error occurred while processing.",
                    'query_result_preview': None,
                })

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False)
        print(f"Conversation with responses saved to {output_path}")
        logger.info("Completed recording conversation with responses.")

    except Exception as e:
        logger.critical("Failed to record conversation with responses", exc_info=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Food-Ops-Bot CLI")
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'replay'], help='Run mode: single, batch, or replay')
    parser.add_argument('--input', type=str, help='Input file for batch or replay mode (CSV with a question column)')
    parser.add_argument('--output', type=str, help='Output file for batch or replay mode')
    args = parser.parse_args()

    if args.mode == 'single':
        main()
    elif args.mode == 'batch':
        # Batch mode: non-interactive
        if args.input:
            # Read questions from input CSV
            import pandas as pd
            input_df = pd.read_csv(args.input)
            if 'question' not in input_df.columns:
                print(f"Input file {args.input} must contain a 'question' column.")
                exit(1)
            questions = input_df['question'].tolist()
            # Prepare output
            output_path = args.output or f"batch_results_{int(time.time())}.csv"
            results = []
            for q in questions:
                user_id = str(uuid.uuid4())
                try:
                    result = process_single_question((user_id, q, None))
                    results.append({
                        'question': q,
                        'query': result.get('query'),
                        'response': result.get('conversation_response'),
                        'rows': result.get('rows'),
                        'time_taken_sec': result.get('time_taken_sec')
                    })
                except Exception as e:
                    results.append({
                        'question': q,
                        'query': None,
                        'response': f"Error: {e}",
                        'rows': None,
                        'time_taken_sec': None
                    })
            # Save results to output CSV
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"Batch results saved to {output_path}")
        else:
            multiprocessor()
    elif args.mode == 'replay':
        if args.input:
            record_conversation_with_responses(args.input, args.output or "conversation_with_responses.csv")
        else:
            csv_path = input("Enter path to conversation CSV: ").strip()
            record_conversation_with_responses(csv_path)
    else:
        # Interactive fallback
        mode = input("Choose mode (single / batch / replay): ").strip().lower()
        if mode == "single":
            main()
        elif mode == "batch":
            multiprocessor()
        elif mode == "replay":
            csv_path = input("Enter path to conversation CSV: ").strip()
            record_conversation_with_responses(csv_path)
        else:
            print("Invalid mode. Please enter 'single', 'batch', or 'replay'.")
