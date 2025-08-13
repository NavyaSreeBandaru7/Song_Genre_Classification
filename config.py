#!/usr/bin/env python3
"""
Configuration Settings for Spotify Analysis Platform
===================================================

Centralized configuration management for all analysis parameters,
model settings, and application preferences.

Author: Data Science Team
Version: 1.0.0
"""

import os
from pathlib import Path

# Base configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
STATIC_DIR = BASE_DIR / 'static'
UPLOAD_DIR = DATA_DIR / 'uploads'
OUTPUT_DIR = BASE_DIR / 'output'

# Create directories if they don't exist
for directory in [DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR]:
    directory.mkdir(exist_ok=True)

# Application Settings
class AppConfig:
    """Flask application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = str(UPLOAD_DIR)
    
    # Database settings (if needed)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///spotify_analysis.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Security settings
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600

# Data Processing Configuration
class DataConfig:
    """Data processing and preprocessing settings"""
    
    # File processing
    ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.json'}
    ENCODING = 'utf-8'
    
    # Missing value handling
    MISSING_VALUE_STRATEGY = 'intelligent'  # 'intelligent', 'simple', 'knn'
    MISSING_THRESHOLD = 0.5  # Remove columns with >50% missing values
    
    # Outlier detection
    OUTLIER_METHOD = 'iqr'  # 'iqr', 'zscore', 'isolation'
    OUTLIER_THRESHOLD = 1.5
    OUTLIER_REMOVAL = False  # Whether to remove outliers automatically
    
    # Feature engineering
    FEATURE_ENGINEERING = True
    AUTO_FEATURE_SELECTION = True
    CORRELATION_THRESHOLD = 0.95  # Remove highly correlated features
    
    # Scaling
    SCALING_METHOD = 'standard'  # 'standard', 'minmax', 'robust'
    SCALE_FEATURES = True

# Machine Learning Configuration
class MLConfig:
    """Machine learning model settings"""
    
    # General ML settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    N_SPLITS = 5  # For cross-validation
    
    # Clustering settings
    CLUSTERING_ALGORITHMS = ['kmeans', 'hierarchical', 'dbscan']
    KMEANS_CLUSTERS_RANGE = range(2, 11)
    KMEANS_INIT = 'k-means++'
    KMEANS_N_INIT = 10
    
    # Classification settings
    CLASSIFICATION_ALGORITHMS = ['random_forest', 'svm', 'gradient_boosting']
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = None
    RF_MIN_SAMPLES_SPLIT = 2
    
    # Model evaluation
    SCORING_METRICS = ['accuracy', 'precision', 'recall', 'f1']
    CROSS_VALIDATION = True
    
    # Feature importance
    FEATURE_IMPORTANCE_THRESHOLD = 0.01
    MAX_FEATURES_DISPLAY = 20

# NLP Configuration
class NLPConfig:
    """Natural Language Processing settings"""
    
    # Text processing
    MIN_WORD_LENGTH = 2
    MAX_WORD_LENGTH = 50
    STOP_WORDS = 'english'
    STEMMING = True
    LEMMATIZATION = False
    
    # Sentiment analysis
    SENTIMENT_ANALYZER = 'vader'  # 'vader', 'textblob'
    SENTIMENT_THRESHOLD = 0.1
    
    # Topic modeling
    N_TOPICS = 5
    TOPIC_MODEL = 'lda'  # 'lda', 'nmf'
    
    # Word cloud settings
    WORDCLOUD_MAX_WORDS = 100
    WORDCLOUD_MIN_FONT_SIZE = 10
    WORDCLOUD_MAX_FONT_SIZE = 100
    WORDCLOUD_BACKGROUND = 'white'

# Visualization Configuration
class VizConfig:
    """Visualization and plotting settings"""
    
    # General plotting
    STYLE = 'seaborn-v0_8'
    COLOR_PALETTE = 'husl'
    FIGURE_SIZE = (12, 8)
    DPI = 300
    SAVE_FORMAT = 'png'
    
    # Interactive plots
    PLOTLY_THEME = 'plotly_white'
    PLOTLY_COLOR_SCALE = 'viridis'
    
    # Specific plot settings
    CORRELATION_HEATMAP = {
        'cmap': 'coolwarm',
        'center': 0,
        'square': True,
        'linewidths': 0.5,
        'annot': True
    }
    
    DISTRIBUTION_PLOTS = {
        'kde': True,
        'bins': 30,
        'alpha': 0.7
    }
    
    SCATTER_PLOTS = {
        'alpha': 0.6,
        'size': 50
    }

# Audio Feature Configuration
class AudioConfig:
    """Audio feature analysis settings"""
    
    # Standard Spotify audio features
    AUDIO_FEATURES = [
        'danceability', 'energy', 'valence', 'acousticness',
        'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness'
    ]
    
    # Feature ranges (for validation)
    FEATURE_RANGES = {
        'danceability': (0, 1),
        'energy': (0, 1),
        'valence': (0, 1),
        'acousticness': (0, 1),
        'instrumentalness': (0, 1),
        'liveness': (0, 1),
        'speechiness': (0, 1),
        'tempo': (50, 250),
        'loudness': (-60, 5),
        'duration_ms': (10000, 900000),  # 10 seconds to 15 minutes
        'popularity': (0, 100)
    }
    
    # Feature categories
    MOOD_FEATURES = ['valence', 'energy', 'danceability']
    ACOUSTIC_FEATURES = ['acousticness', 'instrumentalness', 'liveness']
    TECHNICAL_FEATURES = ['tempo', 'loudness', 'duration_ms']
    
    # Genre classification features
    GENRE_FEATURES = AUDIO_FEATURES + ['popularity', 'duration_ms']

# Performance Configuration
class PerformanceConfig:
    """Performance optimization settings"""
    
    # Memory management
    MAX_MEMORY_USAGE = '4GB'
    CHUNK_SIZE = 10000  # For large file processing
    
    # Parallel processing
    N_JOBS = -1  # Use all available cores
    PARALLEL_BACKEND = 'threading'
    
    # Caching
    ENABLE_CACHING = True
    CACHE_DIR = BASE_DIR / '.cache'
    CACHE_TIMEOUT = 3600  # 1 hour
    
    # Database optimization
    DB_POOL_SIZE = 5
    DB_POOL_TIMEOUT = 30

# Logging Configuration
class LogConfig:
    """Logging configuration"""
    
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = BASE_DIR / 'logs' / 'spotify_analysis.log'
    
    # Create logs directory
    LOG_FILE.parent.mkdir(exist_ok=True)
    
    # Rotating file handler settings
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5

# API Configuration
class APIConfig:
    """API and external service settings"""
    
    # Spotify API (if integrating with live data)
    SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
    SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')
    SPOTIFY_REDIRECT_URI = os.environ.get('SPOTIFY_REDIRECT_URI', 'http://localhost:5000/callback')
    
    # Rate limiting
    RATE_LIMIT = '100/hour'
    RATE_LIMIT_STORAGE = 'memory'
    
    # Request settings
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 1

# Security Configuration
class SecurityConfig:
    """Security and privacy settings"""
    
    # Data privacy
    ANONYMIZE_DATA = False
    REMOVE_PII = True
    
    # File upload security
    SCAN_UPLOADS = True
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_MIME_TYPES = ['text/csv', 'application/csv', 'application/json']
    
    # CORS settings
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5000']
    CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE']

# Environment-specific configurations
class DevelopmentConfig(AppConfig):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False
    
class ProductionConfig(AppConfig):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
class TestingConfig(AppConfig):
    """Testing environment configuration"""
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False

# Configuration factory
def get_config(env=None):
    """
    Get configuration based on environment
    
    Parameters:
    -----------
    env : str
        Environment name ('development', 'production', 'testing')
        
    Returns:
    --------
    Configuration class
    """
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    return config_map.get(env, DevelopmentConfig)

# Validation functions
def validate_audio_features(data):
    """
    Validate audio features are within expected ranges
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to validate
        
    Returns:
    --------
    dict : Validation results
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    for feature, (min_val, max_val) in AudioConfig.FEATURE_RANGES.items():
        if feature in data.columns:
            out_of_range = (data[feature] < min_val) | (data[feature] > max_val)
            if out_of_range.any():
                count = out_of_range.sum()
                percentage = (count / len(data)) * 100
                
                if percentage > 10:  # More than 10% out of range
                    validation_results['errors'].append(
                        f"{feature}: {count} values ({percentage:.1f}%) out of range [{min_val}, {max_val}]"
                    )
                    validation_results['valid'] = False
                else:
                    validation_results['warnings'].append(
                        f"{feature}: {count} values ({percentage:.1f}%) out of range [{min_val}, {max_val}]"
                    )
    
    return validation_results

def get_feature_config(feature_type):
    """
    Get configuration for specific feature type
    
    Parameters:
    -----------
    feature_type : str
        Type of features ('audio', 'mood', 'acoustic', 'technical')
        
    Returns:
    --------
    list : List of feature names
    """
    config_map = {
        'audio': AudioConfig.AUDIO_FEATURES,
        'mood': AudioConfig.MOOD_FEATURES,
        'acoustic': AudioConfig.ACOUSTIC_FEATURES,
        'technical': AudioConfig.TECHNICAL_FEATURES,
        'genre': AudioConfig.GENRE_FEATURES
    }
    
    return config_map.get(feature_type, [])

# Export commonly used configurations
__all__ = [
    'AppConfig', 'DataConfig', 'MLConfig', 'NLPConfig', 'VizConfig',
    'AudioConfig', 'PerformanceConfig', 'LogConfig', 'APIConfig',
    'SecurityConfig', 'get_config', 'validate_audio_features',
    'get_feature_config'
]
