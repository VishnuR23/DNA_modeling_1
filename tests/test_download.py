"""Offline defaults and integrity checks must hold before data processing."""
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from tito_repro.data.download import download_alanine


def download_config(tmp_path: Path, allow_network: bool = False):
    return OmegaConf.create({"download": {
        "directory": str(tmp_path / "cache"), "files": ["sample.xtc"],
        "base_url": "https://example.invalid", "timeout_seconds": 1,
        "allow_network": allow_network,
        "sha256": {"sample.xtc": hashlib.sha256(b"expected").hexdigest()},
    }})


def test_missing_cache_never_starts_network(tmp_path):
    cfg = download_config(tmp_path)
    with patch("tito_repro.data.download.subprocess.run") as run:
        with pytest.raises(FileNotFoundError, match="download.allow_network=true"):
            download_alanine(cfg, tmp_path)
        run.assert_not_called()
    assert not Path(cfg.download.directory).exists()


def test_corrupt_cache_is_not_redownloaded(tmp_path):
    cfg = download_config(tmp_path)
    root = Path(cfg.download.directory)
    root.mkdir()
    (root / "sample.xtc").write_bytes(b"corrupt")
    with patch("tito_repro.data.download.subprocess.run") as run:
        with pytest.raises(ValueError, match="cached checksum"):
            download_alanine(cfg, tmp_path)
        run.assert_not_called()


def test_corrupt_download_does_not_poison_cache(tmp_path):
    cfg = download_config(tmp_path, allow_network=True)

    def fake_download(command, **kwargs):
        Path(command[-1]).write_bytes(b"corrupt")

    with patch("tito_repro.data.download.subprocess.run", side_effect=fake_download) as run:
        with pytest.raises(ValueError, match="Downloaded checksum"):
            download_alanine(cfg, tmp_path)
        run.assert_called_once()
    assert list(Path(cfg.download.directory).iterdir()) == []
