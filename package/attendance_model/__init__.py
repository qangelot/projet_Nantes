import logging

from attendance_model.config.core import PACKAGE_ROOT, config

# only add NullHandler to library’s loggers. Indeed the configuration
# of handlers is the prerogative of the application developer who uses your
# library. And knows their target audience.
logging.getLogger(config.app_config.package_name).addHandler(logging.NullHandler())


with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
