import nltk
import logging

logger = logging.getLogger(__name__)
nltk.download('punkt_tab')

def download_nltk_data():
    required_data = ['punkt', 'stopwords']
    for resource in required_data:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logger.info(f"NLTK resource '{resource}' already present.")
        except LookupError:
            logger.warning(f"NLTK resource '{resource}' not found. Downloading...")
            try:
                nltk.download(resource)
                logger.info(f"NLTK resource '{resource}' downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download NLTK resource '{resource}': {e}")