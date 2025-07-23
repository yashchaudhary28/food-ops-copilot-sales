"""
Unified Configuration Loader for Food-Ops-Bot

This module loads configuration from the unified_config.yaml file and provides
a clean interface for all application components. This makes the entire codebase
configurable through a single YAML file.

Usage:
    from config_loader import Config
    
    # Access any configuration value
    project_id = Config.gcp.project_id
    table_name = Config.bigquery.target_table.table_id
    model_name = Config.models.primary.name
"""

import os
import yaml
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GCPConfig:
    """GCP-related configuration"""
    project_id: str
    region: str
    service_account_key_path: str
    vertex_ai_location: str


@dataclass
class BigQueryTableConfig:
    """BigQuery table configuration"""
    project_id: str
    dataset_id: str
    table_id: str
    schema_file: str
    description: str


@dataclass
class BigQueryConfig:
    """BigQuery-related configuration"""
    location: str
    target_table: BigQueryTableConfig
    export_dataset_id: str
    export_table_id: str


@dataclass
class ModelConfig:
    """AI model configuration"""
    name: str
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 1


@dataclass
class ModelsConfig:
    """All AI models configuration"""
    primary: ModelConfig
    reflection: ModelConfig
    embedding_name: str


@dataclass
class ApplicationConfig:
    """Application-specific configuration"""
    name: str
    description: str
    schema_config_path: str
    few_shot_examples_path: str
    embedding_store_path: str
    logging_config_path: str
    ui_title: str
    ui_description: str
    show_sql_toggle: bool
    show_metadata_toggle: bool
    max_display_rows: int
    enable_reflection: bool
    enable_few_shot: bool
    enable_chat_history: bool
    max_history_length: int


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    environment: str
    port: int
    cloud_run_service_name: str
    cloud_run_region: str
    cloud_run_memory: str
    cloud_run_cpu: str
    cloud_run_max_instances: int


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str
    format: str
    enable_cloud_logging: bool


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_auth: bool
    allowed_origins: list


@dataclass
class FeaturesConfig:
    """Feature flags configuration"""
    enable_batch_processing: bool
    enable_export_to_bigquery: bool
    enable_query_caching: bool
    enable_usage_analytics: bool


class ConfigLoader:
    """
    Main configuration loader class that reads from unified_config.yaml
    and provides structured access to all configuration values.
    """
    
    def __init__(self, config_file: str = "unified_config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_file: Path to the YAML configuration file
        """
        self.config_file = config_file
        self._config_data = None
        self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file with environment variable substitution"""
        config_path = Path(__file__).parent / self.config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as file:
                config_content = file.read()
                
            # Substitute environment variables in the format ${VAR_NAME}
            config_content = self._substitute_env_vars(config_content)
            
            self._config_data = yaml.safe_load(config_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in the format ${VAR_NAME}"""
        def replace_env_var(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(f"Environment variable {var_name} is not set")
            return env_value
        
        # Replace ${VAR_NAME} with environment variable values
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, content)
    
    @property
    def gcp(self) -> GCPConfig:
        """Get GCP configuration"""
        gcp_config = self._config_data['gcp']
        return GCPConfig(
            project_id=gcp_config['project_id'],
            region=gcp_config['region'],
            service_account_key_path=gcp_config.get('service_account_key_path', ''),
            vertex_ai_location=gcp_config['vertex_ai']['location']
        )
    
    @property
    def bigquery(self) -> BigQueryConfig:
        """Get BigQuery configuration"""
        bq_config = self._config_data['bigquery']
        target_table = bq_config['target_table']
        
        return BigQueryConfig(
            location=bq_config['location'],
            target_table=BigQueryTableConfig(
                project_id=target_table['project_id'],
                dataset_id=target_table['dataset_id'],
                table_id=target_table['table_id'],
                schema_file=target_table['schema_file'],
                description=target_table['description']
            ),
            export_dataset_id=bq_config['export']['dataset_id'],
            export_table_id=bq_config['export']['table_id']
        )
    
    @property
    def models(self) -> ModelsConfig:
        """Get AI models configuration"""
        models_config = self._config_data['models']
        
        return ModelsConfig(
            primary=ModelConfig(
                name=models_config['primary']['name'],
                temperature=models_config['primary']['temperature'],
                top_p=models_config['primary']['top_p'],
                top_k=models_config['primary']['top_k']
            ),
            reflection=ModelConfig(
                name=models_config['reflection']['name'],
                temperature=models_config['reflection']['temperature'],
                top_p=models_config['reflection']['top_p'],
                top_k=models_config['reflection']['top_k']
            ),
            embedding_name=models_config['embedding']['name']
        )
    
    @property
    def application(self) -> ApplicationConfig:
        """Get application configuration"""
        app_config = self._config_data['application']
        files = app_config['files']
        ui = app_config['ui']
        processing = app_config['processing']
        
        return ApplicationConfig(
            name=app_config['name'],
            description=app_config['description'],
            schema_config_path=files['schema_config'],
            few_shot_examples_path=files['few_shot_examples'],
            embedding_store_path=files['embedding_store'],
            logging_config_path=files['logging_config'],
            ui_title=ui['title'],
            ui_description=ui['description'],
            show_sql_toggle=ui['show_sql_toggle'],
            show_metadata_toggle=ui['show_metadata_toggle'],
            max_display_rows=ui['max_display_rows'],
            enable_reflection=processing['enable_reflection'],
            enable_few_shot=processing['enable_few_shot'],
            enable_chat_history=processing['enable_chat_history'],
            max_history_length=processing['max_history_length']
        )
    
    @property
    def deployment(self) -> DeploymentConfig:
        """Get deployment configuration"""
        deploy_config = self._config_data['deployment']
        cloud_run = deploy_config['cloud_run']
        
        return DeploymentConfig(
            environment=deploy_config['environment'],
            port=deploy_config['port'],
            cloud_run_service_name=cloud_run['service_name'],
            cloud_run_region=cloud_run['region'],
            cloud_run_memory=cloud_run['memory'],
            cloud_run_cpu=cloud_run['cpu'],
            cloud_run_max_instances=cloud_run['max_instances']
        )
    
    @property
    def logging(self) -> LoggingConfig:
        """Get logging configuration"""
        log_config = self._config_data['logging']
        
        return LoggingConfig(
            level=log_config['level'],
            format=log_config['format'],
            enable_cloud_logging=log_config['enable_cloud_logging']
        )
    
    @property
    def security(self) -> SecurityConfig:
        """Get security configuration"""
        sec_config = self._config_data['security']
        
        return SecurityConfig(
            enable_auth=sec_config['enable_auth'],
            allowed_origins=sec_config['allowed_origins']
        )
    
    @property
    def features(self) -> FeaturesConfig:
        """Get feature flags configuration"""
        feat_config = self._config_data['features']
        
        return FeaturesConfig(
            enable_batch_processing=feat_config['enable_batch_processing'],
            enable_export_to_bigquery=feat_config['enable_export_to_bigquery'],
            enable_query_caching=feat_config['enable_query_caching'],
            enable_usage_analytics=feat_config['enable_usage_analytics']
        )
    
    def get_full_table_name(self) -> str:
        """Get the fully qualified table name for BigQuery"""
        bq = self.bigquery
        return f"{bq.target_table.project_id}.{bq.target_table.dataset_id}.{bq.target_table.table_id}"
    
    def get_export_table_name(self) -> str:
        """Get the fully qualified export table name"""
        bq = self.bigquery
        return f"{self.gcp.project_id}.{bq.export_dataset_id}.{bq.export_table_id}"
    
    def initialize_gcp_services(self):
        """Initialize GCP services with the configured settings"""
        try:
            # Set up authentication if service account key is provided
            if self.gcp.service_account_key_path and os.path.exists(self.gcp.service_account_key_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.gcp.service_account_key_path
                print(f"✅ Using service account key: {self.gcp.service_account_key_path}")
            else:
                print("ℹ️  Using default application credentials")
            
            # Initialize Vertex AI
            try:
                import vertexai
                vertexai.init(project=self.gcp.project_id, location=self.gcp.vertex_ai_location)
                print(f"✅ Vertex AI initialized for project '{self.gcp.project_id}' in region '{self.gcp.vertex_ai_location}'")
            except ImportError:
                print("⚠️  vertexai package not found. Vertex AI features will be disabled.")
                
        except Exception as e:
            print(f"❌ Error initializing GCP services: {str(e)}")


# Global configuration instance
Config = ConfigLoader()

# Initialize GCP services when the module is imported
Config.initialize_gcp_services()


# Backward compatibility aliases for existing code
class AppConfig:
    """
    Backward compatibility class that maps to the new unified configuration.
    This allows existing code to work without changes while using the new config system.
    """
    
    @property
    def TABLE_NAME(self) -> str:
        return Config.get_full_table_name()
    
    @property
    def SCHEMA_PATH(self) -> str:
        return Config.application.schema_config_path
    
    @property
    def FEWSHOT_INFO_PATH(self) -> str:
        return Config.application.few_shot_examples_path
    
    @property
    def MODEL_NAME(self) -> str:
        return Config.models.primary.name
    
    @property
    def REFLECTION_MODEL_NAME(self) -> str:
        return Config.models.reflection.name
    
    @property
    def EXPORT_DATASET(self) -> str:
        return Config.bigquery.export_dataset_id
    
    @property
    def EXPORT_TABLE(self) -> str:
        return Config.bigquery.export_table_id


# Create a global instance for backward compatibility
AppConfig = AppConfig()


# Backward compatibility class for existing GCPConfig usage
class BackwardCompatibleGCPConfig:
    """Backward compatibility for existing GCPConfig usage"""
    
    @property
    def PROJECT_ID(self) -> str:
        return Config.gcp.project_id
    
    @property
    def VERTEX_AI_LOCATION(self) -> str:
        return Config.gcp.vertex_ai_location


# Create a global instance for backward compatibility
GCPConfigCompat = BackwardCompatibleGCPConfig()
