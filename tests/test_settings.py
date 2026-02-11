"""Tests for the settings module."""
import os
from unittest import mock

from datafaker.settings import (
    Settings,
    SettingsError,
    get_destination_dsn,
    get_source_dsn,
)
from tests.utils import DatafakerTestCase


class TestSettings(DatafakerTestCase):
    """Tests for the Settings class."""

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_minimal_settings(self) -> None:
        """Test the minimal settings."""
        settings = Settings()
        self.assertIsNone(settings.src_dsn)
        self.assertIsNone(settings.src_schema)

        self.assertIsNone(settings.dst_dsn)
        self.assertIsNone(settings.dst_schema)

    def test_maximal_settings(self) -> None:
        """Test the full settings."""
        Settings(
            src_dsn="postgresql://user:password@host:port/db_name?sslmode=require",
            src_schema="dst_schema",
            dst_dsn="postgresql://user:password@host:port/db_name?sslmode=require",
            dst_schema="src_schema",
            # To stop any local .env files influencing the test
            # The mypy ignore can be removed once we upgrade to pydantic 2.
            _env_file=None,  # type: ignore[call-arg]
        )

    def test_validation(self) -> None:
        """Schema settings aren't compatible with MariaDB."""
        with self.assertRaises(SettingsError):
            Settings(
                src_dsn="mariadb+pymysql://myuser@localhost:3306/testdb", src_schema=""
            )

        with self.assertRaises(SettingsError):
            Settings(
                dst_dsn="mariadb+pymysql://myuser@localhost:3306/testdb", dst_schema=""
            )

    @mock.patch("datafaker.settings.get_settings")
    def test_get_destination_dsn_raises_if_no_dsn(
        self, mock_get_settings: mock.MagicMock
    ) -> None:
        """Test that get_destination_dsn raises if dst DSN is missing."""
        mock_get_settings.return_value = Settings(
            src_dsn="mariadb+pymysql://myuser@localhost:3306/testdb",
            dst_dsn=None,
        )
        with self.assertRaises(SettingsError) as context_manager:
            get_destination_dsn()
        self.assertEqual(context_manager.exception.args[0], "Missing DST_DSN setting")

    @mock.patch("datafaker.settings.get_settings")
    def test_get_source_dsn_raises_if_no_dsn(
        self, mock_get_settings: mock.MagicMock
    ) -> None:
        """Test that get_destination_dsn raises if src DSN is missing."""
        mock_get_settings.return_value = Settings(
            src_dsn=None,
            dst_dsn="mariadb+pymysql://myuser@localhost:3306/testdb",
        )
        with self.assertRaises(SettingsError) as context_manager:
            get_source_dsn()
        self.assertEqual(context_manager.exception.args[0], "Missing SRC_DSN setting")
