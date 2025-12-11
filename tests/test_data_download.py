"""
Tests for the data_download module.

This module tests the download_file and download_wine_datasets functions
to ensure they correctly download files and handle errors appropriately.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_download import download_file, download_wine_datasets, WINE_DATA_URLS


class TestDownloadFile:
    """Tests for the download_file function."""

    def test_download_file_success(self, tmp_path):
        """Test successful file download with mocked response."""
        test_content = b"test,data\n1,2\n3,4"
        output_file = tmp_path / "test.csv"

        with patch("src.data_download.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.content = test_content
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = download_file("https://example.com/data.csv", output_file)

            assert result == output_file
            assert output_file.exists()
            assert output_file.read_bytes() == test_content

    def test_download_file_creates_parent_dirs(self, tmp_path):
        """Test that download_file creates parent directories if needed."""
        test_content = b"test data"
        output_file = tmp_path / "nested" / "dir" / "test.csv"

        with patch("src.data_download.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.content = test_content
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = download_file("https://example.com/data.csv", output_file)

            assert result == output_file
            assert output_file.exists()

    def test_download_file_empty_url_raises_error(self, tmp_path):
        """Test that empty URL raises ValueError."""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            download_file("", tmp_path / "test.csv")

    def test_download_file_empty_path_raises_error(self):
        """Test that empty output path raises ValueError."""
        with pytest.raises(ValueError, match="Output path cannot be empty"):
            download_file("https://example.com/data.csv", "")

    def test_download_file_http_error(self, tmp_path):
        """Test that HTTP errors are raised."""
        output_file = tmp_path / "test.csv"

        with patch("src.data_download.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "404 Not Found"
            )
            mock_get.return_value = mock_response

            with pytest.raises(requests.exceptions.HTTPError):
                download_file("https://example.com/notfound.csv", output_file)

    def test_download_file_timeout(self, tmp_path):
        """Test that timeout errors are raised."""
        output_file = tmp_path / "test.csv"

        with patch("src.data_download.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

            with pytest.raises(requests.exceptions.Timeout):
                download_file("https://example.com/data.csv", output_file, timeout=1)

    def test_download_file_string_path(self, tmp_path):
        """Test that string paths are accepted."""
        test_content = b"test data"
        output_file = str(tmp_path / "test.csv")

        with patch("src.data_download.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.content = test_content
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = download_file("https://example.com/data.csv", output_file)

            assert Path(result).exists()


class TestDownloadWineDatasets:
    """Tests for the download_wine_datasets function."""

    def test_download_wine_datasets_success(self, tmp_path):
        """Test successful download of both wine datasets."""
        red_content = b"fixed acidity;volatile acidity\n7.4;0.7"
        white_content = b"fixed acidity;volatile acidity\n7.0;0.27"

        with patch("src.data_download.download_file") as mock_download:
            mock_download.side_effect = [
                tmp_path / "winequality-red.csv",
                tmp_path / "winequality-white.csv",
            ]
            # Create the files to simulate download
            (tmp_path / "winequality-red.csv").write_bytes(red_content)
            (tmp_path / "winequality-white.csv").write_bytes(white_content)

            red_path, white_path = download_wine_datasets(tmp_path)

            assert red_path.name == "winequality-red.csv"
            assert white_path.name == "winequality-white.csv"
            assert mock_download.call_count == 2

    def test_download_wine_datasets_empty_dir_raises_error(self):
        """Test that empty output directory raises ValueError."""
        with pytest.raises(ValueError, match="Output directory cannot be empty"):
            download_wine_datasets("")

    def test_download_wine_datasets_creates_directory(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_data_dir"

        with patch("src.data_download.download_file") as mock_download:
            mock_download.side_effect = [
                new_dir / "winequality-red.csv",
                new_dir / "winequality-white.csv",
            ]

            download_wine_datasets(new_dir)

            assert new_dir.exists()

    def test_download_wine_datasets_uses_default_urls(self, tmp_path):
        """Test that default UCI URLs are used."""
        with patch("src.data_download.download_file") as mock_download:
            mock_download.side_effect = [
                tmp_path / "winequality-red.csv",
                tmp_path / "winequality-white.csv",
            ]

            download_wine_datasets(tmp_path)

            # Check that the correct URLs were used
            calls = mock_download.call_args_list
            assert WINE_DATA_URLS["red"] in str(calls[0])
            assert WINE_DATA_URLS["white"] in str(calls[1])

    def test_download_wine_datasets_custom_urls(self, tmp_path):
        """Test that custom URLs can be provided."""
        custom_red = "https://custom.url/red.csv"
        custom_white = "https://custom.url/white.csv"

        with patch("src.data_download.download_file") as mock_download:
            mock_download.side_effect = [
                tmp_path / "winequality-red.csv",
                tmp_path / "winequality-white.csv",
            ]

            download_wine_datasets(
                tmp_path, red_url=custom_red, white_url=custom_white
            )

            calls = mock_download.call_args_list
            assert custom_red in str(calls[0])
            assert custom_white in str(calls[1])
