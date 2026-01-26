import logging as _logging

import jax
from absl import logging

def log_for_0(*args, stacklevel=2):
    if jax.process_index() == 0:
        logging.info(*args, stacklevel=stacklevel)

print0 = lambda *args, **kwargs: log_for_0(*args, stacklevel=3)

def log_for_all(msg):
    logging.info(f"[Rank {jax.process_index()}] {msg}")

class ExcludeInfo(_logging.Filter):
    def __init__(self, exclude_files):
        super().__init__()
        self.exclude_files = exclude_files

    def filter(self, record):
        if any(file_name in record.pathname for file_name in self.exclude_files):
            return record.levelno > _logging.INFO
        return True

exclude_files = [
    'orbax/checkpoint/async_checkpointer.py',
    'orbax/checkpoint/abstract_checkpointer.py',
    'orbax/checkpoint/multihost/utils.py',
    'orbax/checkpoint/future.py',
    'orbax/checkpoint/_src/handlers/base_pytree_checkpoint_handler.py',
    'orbax/checkpoint/type_handlers.py',
    'orbax/checkpoint/metadata/checkpoint.py',
    'orbax/checkpoint/metadata/sharding.py',
] + [
    'orbax/checkpoint/checkpointer.py',
    'flax/training/checkpoints.py',
] * jax.process_index()
file_filter = ExcludeInfo(exclude_files)

def supress_checkpt_info():
    logging.get_absl_handler().addFilter(file_filter)


class Emoji:
    HAPPY = "😀"
    THUMBS = "👍"
    YEAH = "🎉"
    ROCKET = "🚀"
    SPARKLES = "✨"
    FIRE = "🔥"
    GOOD = "✅"
    WARNING = "⚠️ "
    ERROR = "❌"
    EYES = "👀"
    TRUCK = "🚛"
    ROBOT = "🤖"
    INFO = "ℹ️ "