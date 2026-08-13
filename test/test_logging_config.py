"""Unit tests for logging_config.py. Pure stdlib -- does not need IMP or JAX."""

import logging
import os
import tempfile
import unittest

from impjax_toymodels.logging_config import configure_logging


class LoggingConfigTests(unittest.TestCase):
    def test_configure_logging_returns_named_logger_at_requested_level(self):
        logger = configure_logging(level=logging.DEBUG)
        self.assertEqual(logger.name, "impjax_toymodels")
        self.assertEqual(logger.level, logging.DEBUG)

    def test_repeated_calls_do_not_duplicate_handlers(self):
        configure_logging()
        configure_logging()
        logger = configure_logging()
        self.assertEqual(len(logger.handlers), 1)

    def test_log_path_adds_file_handler_and_writes_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "run.log")
            logger = configure_logging(log_path=log_path)
            self.assertEqual(len(logger.handlers), 2)
            logger.info("hello from test")
            for handler in logger.handlers:
                handler.flush()
            with open(log_path) as f:
                contents = f.read()
            self.assertIn("hello from test", contents)


if __name__ == "__main__":
    unittest.main()
