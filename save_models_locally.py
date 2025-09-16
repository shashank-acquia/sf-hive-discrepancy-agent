from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Downloads and saves the sentence-transformer model to a local directory.
    This allows it to be copied directly into a Docker image, bypassing
    the need for the container to have internet access to Hugging Face.
    """
    model_name = 'all-MiniLM-L6-v2'
    save_path = f'./{model_name}'  # Save it in a folder with the model's name

    logger.info(f"Downloading sentence-transformer model '{model_name}' to '{save_path}'...")

    try:
        # Initialize the model - this triggers the download
        model = SentenceTransformer(model_name)

        # Explicitly save it to a local path
        model.save(save_path)

        logger.info(f"Successfully downloaded and saved model to '{save_path}'.")
        logger.info("You can now build the Docker image.")
    except Exception as e:
        logger.error(f"Failed to download model '{model_name}': {e}")
        logger.error("Please check your local network connection and proxy settings.")


if __name__ == '__main__':
    main()