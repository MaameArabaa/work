from datetime import datetime
import os

# Create logs folder if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "rag_logs.txt")


def log_message(title, content):
    """
    Logs messages with timestamp, title, and content.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"{title}\n")
        f.write(f"Time: {timestamp}\n")
        f.write("-" * 70 + "\n")
        f.write(str(content))
        f.write("\n")


def log_error(error_message):
    """
    Logs errors separately for debugging.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "!" * 70 + "\n")
        f.write("ERROR\n")
        f.write(f"Time: {timestamp}\n")
        f.write("-" * 70 + "\n")
        f.write(str(error_message))
        f.write("\n")