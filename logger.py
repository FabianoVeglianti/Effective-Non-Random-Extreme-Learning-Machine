import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

logged_messages = []
def log_info_once(message):
    if not message in logged_messages:
        logged_messages.append(message)
        log.info(message)

def error(message):
    log.error(message)

def warning(message):
    log.warning(message)