import json
from datetime import datetime
from collections import defaultdict, deque
from logger_context import call_id_var
from logger_util import setup_logger

logger = setup_logger(__name__)

class ChatHistoryManager:
    def __init__(self, retain_last_n=3):
        self.retain_last_n = retain_last_n
        self._history = defaultdict(lambda: deque(maxlen=self.retain_last_n))
        logger.info(f"ChatHistoryManager initialized with retain_last_n={retain_last_n}")

    def append(self, question: str, info: dict, query: str, conversation_response: str):
        try:
            user_id = call_id_var.get()
            timestamp = datetime.utcnow().isoformat()

            new_data = {
                "user_id": user_id,
                "timestamp": timestamp,
                "question": question or "",
                "info": json.dumps(info or {}, ensure_ascii=False),
                "query": query or "",
                "conversation_response": conversation_response or "",
            }

            self._history[user_id].append(new_data)
            logger.info(f"Appended new history for user_id={user_id} at {timestamp}")
        except Exception as e:
            logger.error(f"Error appending chat history: {e}", exc_info=True)

    def get_last_n(self, n=3) -> str:
        try:
            user_id = call_id_var.get()

            if user_id not in self._history or not self._history[user_id]:
                logger.info(f"No previous history found for user_id={user_id}")
                return "No previous history available for this user."

            last_n = list(self._history[user_id])[-n:]
            logger.info(f"Retrieved last {n} history entries for user_id={user_id}")

            history_strs = []
            for i, record in enumerate(last_n, 1):
                timestamp = record.get("timestamp", "Unknown time")
                question = str(record.get("question", "") or "").strip()
                info = str(record.get("info", "") or "").strip()
                query = str(record.get("query", "") or "").strip()
                response = str(record.get("conversation_response", "") or "").strip()
                formatted = (
                    f"--- Conversation {i} ({timestamp}) ---\n"
                    f"Question {i}:\n{question}\n\n"
                    f"Meta Data for Generating Sql:\n{info}\n\n"
                    f"Generated SQL:\n{query}\n\n"
                    f"Assistant Response:\n{response}\n"
                )
                history_strs.append(formatted)

            return "\n".join(history_strs)

        except Exception as e:
            logger.error(f"Error retrieving chat history for user: {e}", exc_info=True)
            return "Error while retrieving history."
