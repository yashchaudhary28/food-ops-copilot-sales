import chainlit as cl
from chainlit.input_widget import Switch
import pandas as pd
import asyncio
import sys
import os
import warnings
import uuid
import json

from sql_agent import SqlAgent
from reflection_agent import ReflectionAgent
from embedding_model import EmbeddingModel
from chat_history_manager import ChatHistoryManager
from logger_util import setup_logger
import pandas_gbq
from logger_context import call_id_var
from config_loader import Config, AppConfig, GCPConfig

# Suppress asyncio warnings on Windows
if sys.platform == "win32":
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*ProactorEventLoop.*")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configuration is now loaded from unified_config.yaml via config_loader.py
# The AppConfig class is imported from config_loader for backward compatibility

logger = setup_logger(__name__)

def to_bq(df, dataset_id, table_id, if_exists='replace', project_id=None):
    """Export DataFrame to BigQuery using project from config if not specified."""
    project_id = project_id or Config.gcp.project_id
    table_full_id = f"{project_id}.{dataset_id}.{table_id}"
    pandas_gbq.to_gbq(df, 
                     destination_table=table_full_id, 
                     project_id=project_id, 
                     if_exists=if_exists)
    logger.info(f"Data written to {table_full_id} successfully with shape {df.shape}.")
    
def format_df_markdown(df: pd.DataFrame, max_rows=100):
    preview = df.head(max_rows).copy()
    for col in preview.select_dtypes(include='number').columns:
        preview[col] = preview[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, float) else f"{x:,}")
    headers = "| " + " | ".join([f"**{col}**" for col in preview.columns]) + " |"
    separator = "| " + " | ".join(["---"] * len(preview.columns)) + " |"
    rows = "\n".join(["| " + " | ".join(map(str, row)) + " |" for row in preview.values])
    return f"{headers}\n{separator}\n{rows}"

def format_metadata(info):
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except Exception:
            return f"```\n{info}\n```"
    if isinstance(info, dict):
        lines = ["| Key | Value |", "|---|---|"]
        for k, v in info.items():
            lines.append(f"| `{k}` | `{v}` |")
        return "\n".join(lines)
    return f"```\n{str(info)}\n```"

@cl.on_chat_start
async def start():
    cl.user_session.set("debug", False)
    cl.user_session.set("query", None)
    cl.user_session.set("info", None)
    cl.user_session.set("df", None)
    cl.user_session.set("full_df", None)
    cl.user_session.set("user_id", str(uuid.uuid4()))
    cl.user_session.set("show_sql", False)
    cl.user_session.set("show_meta", False)
    cl.user_session.set("sql_msg", None)
    cl.user_session.set("meta_msg", None)
    cl.user_session.set("action_buttons_msg", None)

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
    
    embedding_model = EmbeddingModel(project_id=Config.gcp.project_id)
    cl.user_session.set("embedding_store", embedding_store)
    cl.user_session.set("embedding_model", embedding_model)
    cl.user_session.set("chat_history_manager", ChatHistoryManager())

    await cl.ChatSettings([
        Switch(id="debug_mode", label="🐞 Debug Mode", initial=False)
    ]).send()

    welcome_message = """👋 ***Welcome to the Food-Ops-Copilot!***

I'm ready to answer your questions about restaurant data and food operations.
Type your question below to get started!

---

💡 *Example questions:*
- What are the top 10 restaurants by rating in Bangalore?
- Show me restaurants with the highest number of votes by city
- What's the average cost for two people across different cuisines?
- Which restaurants offer online delivery and have ratings above 4.0?
- What are the most popular cuisines in different cities?

💡 *Use the Debug Mode toggle in settings to show/hide SQL queries!*"""

    await cl.Message(content=welcome_message).send()

@cl.on_settings_update
async def update_settings(settings):
    cl.user_session.set("debug", settings.get("debug_mode", False))

@cl.on_message
async def handle_message(message: cl.Message):
    try:
        prev_task = cl.user_session.get("processing_task")
        if prev_task and not prev_task.done():
            prev_task.cancel()
            logger.info("Previous backend task cancelled due to new user input.")

        task = asyncio.create_task(process_question(message))
        cl.user_session.set("processing_task", task)
        await task

    except asyncio.CancelledError:
        logger.info("Backend processing was cancelled.")
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        await cl.Message(f"❌ Error: {str(e)}").send()

async def process_question(message: cl.Message):
    question = message.content.strip()
    if not question:
        await cl.Message("⚠️ Please enter a question.").send()
        return

    user_id = cl.user_session.get("user_id")
    call_id_var.set(user_id)

    generate_msg = await cl.Message("⏳ Processing your question...").send()

    embedding_store = cl.user_session.get("embedding_store")
    embedding_model = cl.user_session.get("embedding_model")
    chat_history_manager = cl.user_session.get("chat_history_manager")

    reflection_agent = ReflectionAgent(
        model_name=AppConfig.REFLECTION_MODEL_NAME,
        temperature=0.0,
        top_p=0.0,
        top_k=1
    )

    agent = SqlAgent(
        question=question,
        table_name=AppConfig.TABLE_NAME,
        schema_path=AppConfig.SCHEMA_PATH,
        fewshotInfo_path=AppConfig.FEWSHOT_INFO_PATH,
        model_name=AppConfig.MODEL_NAME,
        reflection_model_name=AppConfig.REFLECTION_MODEL_NAME,
        project_id=Config.gcp.project_id,
        reflection_agent=reflection_agent,
        embedding_model=embedding_model,
        chat_history_manager=chat_history_manager,
        embedding_store=embedding_store
    )

    await asyncio.sleep(0.5)
    generate_msg.content = "🔍 Extracting metadata and generating SQL query..."
    await generate_msg.update()

    info, query, conversation_response, df = await asyncio.to_thread(agent.run)

    generate_msg.content = "✅ Processing complete!"
    await generate_msg.update()

    cl.user_session.set("query", query)
    cl.user_session.set("info", info)
    cl.user_session.set("df", df)
    cl.user_session.set("full_df", df)
    cl.user_session.set("show_sql", False)
    cl.user_session.set("show_meta", False)
    cl.user_session.set("sql_msg", None)
    cl.user_session.set("meta_msg", None)
    cl.user_session.set("action_buttons_msg", None)

    if conversation_response:
        await cl.Message(f"🧠 {conversation_response}").send()

    question_entry_df = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "user_id": user_id,
        "question": question,
        "query": query
        # "metadata": None,
        # "conversation_response": conversation_response
    }])
    to_bq(question_entry_df, 
          AppConfig.EXPORT_DATASET, 
          AppConfig.EXPORT_TABLE, 
          if_exists="append")

    if query:
        if df is not None and not df.empty:
            df_preview = format_df_markdown(df)
            await cl.Message(f"📊 **Results:**\n\n{df_preview}").send()

            csv_path = "query_results.csv"
            df.to_csv(csv_path, index=False)
            await cl.Message(
                content="📥 Download results:",
                elements=[cl.File(name="query_results.csv", path=csv_path)]
            ).send()
        elif df is not None:
            await cl.Message("ℹ️ Query ran successfully but returned no data.").send()
        else:
            await cl.Message("⚠️ Query execution failed, even after reflection.").send()
    else:
        await cl.Message("❌ Failed to generate a SQL query.").send()

    await send_action_buttons()

async def send_action_buttons():
    show_sql = cl.user_session.get("show_sql", False)
    show_meta = cl.user_session.get("show_meta", False)

    actions = [
        cl.Action(
            name="toggle_sql",
            value="sql",
            label="🙈 Hide SQL" if show_sql else "🧾 View SQL",
            payload={"action_type": "toggle_sql"}
        ),
        cl.Action(
            name="toggle_meta",
            value="meta",
            label="🙈 Hide Metadata" if show_meta else "📌 Metadata Debug",
            payload={"action_type": "toggle_meta"}
        )
    ]

    action_buttons_msg = await cl.Message(
        content="🔧 Use buttons below for more actions.",
        actions=actions
    ).send()
    cl.user_session.set("action_buttons_msg", action_buttons_msg)

@cl.action_callback("toggle_sql")
async def toggle_sql(action):
    show_sql = not cl.user_session.get("show_sql", False)
    cl.user_session.set("show_sql", show_sql)

    query = cl.user_session.get("query")

    if show_sql and query:
        sql_msg = cl.user_session.get("sql_msg")
        content = f"🧾 **SQL Query:**\n```sql\n{query}\n```"
        if sql_msg is None:
            sql_msg = await cl.Message(content=content, author="Assistant").send()
            cl.user_session.set("sql_msg", sql_msg)
        else:
            sql_msg.content = content
            await sql_msg.update()
    elif show_sql and not query:
        await cl.Message("⚠️ No SQL query available.").send()
    else:
        sql_msg = cl.user_session.get("sql_msg")
        if sql_msg:
            sql_msg.content = "🔒 **SQL Query:** *[Hidden - Click 'View SQL' to show]*"
            await sql_msg.update()

@cl.action_callback("toggle_meta")
async def toggle_meta(action):
    show_meta = not cl.user_session.get("show_meta", False)
    cl.user_session.set("show_meta", show_meta)

    info = cl.user_session.get("info")

    if show_meta and info:
        info_copy = info.copy() if isinstance(info, dict) else info
        if isinstance(info_copy, dict):
            info_copy.pop("relevant_columns_info", None)

        formatted_info = format_metadata(info_copy)
        content = f"📌 **Metadata:**\n{formatted_info}"

        meta_msg = cl.user_session.get("meta_msg")
        if meta_msg is None:
            meta_msg = await cl.Message(content=content, author="Assistant").send()
            cl.user_session.set("meta_msg", meta_msg)
        else:
            meta_msg.content = content
            await meta_msg.update()
    elif show_meta and not info:
        await cl.Message("⚠️ No metadata available.").send()
    else:
        meta_msg = cl.user_session.get("meta_msg")
        if meta_msg:
            meta_msg.content = "📌 **Metadata:** *[Hidden - Click 'Metadata Debug' to show]*"
            await meta_msg.update()
