import ast
import pandas as pd
from tqdm import tqdm
import vertexai
from vertexai.language_models import TextEmbeddingModel
from typing import Optional
from logger_util import setup_logger
from logger_context import call_id_var
from config_loader import Config

logger = setup_logger("embedding_model")

class EmbeddingModel:
    def __init__(self, project_id: Optional[str] = None, region: Optional[str] = None):
        # Use config values if parameters not provided
        self.project_id = project_id or Config.gcp.project_id
        self.region = region or Config.gcp.vertex_ai_location
        self._model = None

        if not self.project_id or not self.region:
            raise ValueError("Both project_id and region must be provided either through parameters or config.")

        try:
            vertexai.init(project=self.project_id, location=self.region)
            logger.info(f"Vertex AI initialized for project '{self.project_id}' in region '{self.region}'")
        except Exception as e:
            logger.critical("Failed to initialize Vertex AI.", exc_info=True)
            raise

    def get_model(self) -> TextEmbeddingModel:
        if self._model is None:
            try:
                self._model = TextEmbeddingModel.from_pretrained(Config.models.embedding_name)
                logger.info("Successfully loaded Gemini embedding model.")
            except Exception as e:
                logger.critical("Failed to load Gemini embedding model.", exc_info=True)
                raise
        return self._model

    def create_embeddings(self, csv_path: str, question_col: str = "question") -> pd.DataFrame:
        try:
            data = pd.read_csv(csv_path)
        except Exception as e:
            logger.critical(f"Failed to read CSV at: {csv_path}", exc_info=True)
            raise

        if question_col not in data.columns:
            raise ValueError(f"Column '{question_col}' not found in CSV.")

        model = self.get_model()
        embeddings_list = []

        logger.info(f"Creating embeddings for {len(data)} questions.")
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Embedding rows"):
            question = str(row[question_col]).strip()
            try:
                embeddings = model.get_embeddings([question])  # Must be batch of 1
                vector = embeddings[0].values
            except Exception as e:
                logger.warning(f"Failed to embed question: '{question}'", exc_info=True)
                vector = []  # or [0.0]*768 depending on downstream use
            embeddings_list.append(vector)

        data["embedding"] = embeddings_list

        try:
            data.to_csv(csv_path, index=False)
            logger.info(f"Embeddings saved to: {csv_path}")
        except Exception as e:
            logger.error("Failed to write embeddings to CSV.", exc_info=True)
            raise

        return data

    @staticmethod
    def parse_embeddings_column(data: pd.DataFrame, embedding_col: str = "embedding") -> pd.DataFrame:
        """Safely converts stringified embeddings back to list of floats."""
        try:
            data[embedding_col] = data[embedding_col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        except Exception as e:
            logger.error("Failed to parse embeddings column.", exc_info=True)
            raise
        return data
