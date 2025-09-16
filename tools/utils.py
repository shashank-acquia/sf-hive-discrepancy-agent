import nltk
import logging
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_nltk_data():
    """
    Downloads NLTK resources required by the application.
    This function handles SSL errors and ensures the correct packages are present.
    """
    packages = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords'
    }

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    logger.info("Verifying NLTK data...")
    for pkg_name, pkg_path in packages.items():
        try:
            nltk.data.find(pkg_path)
            logger.info(f"NLTK package '{pkg_name}' is available.")
        except LookupError:
            logger.warning(f"NLTK package '{pkg_name}' not found. Downloading...")
            try:
                nltk.download(pkg_name, quiet=True)
                logger.info(f"Successfully downloaded NLTK package '{pkg_name}'.")
            except Exception as e:
                logger.error(f"Failed to download NLTK package '{pkg_name}': {e}")
                raise RuntimeError(f"Could not download essential NLTK package: {pkg_name}")


if __name__ == '__main__':
    download_nltk_data()