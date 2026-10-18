import os

from openoe.core.config import load_config


def test_quoted_dotenv_credentials_and_export_are_loaded_without_quotes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    (tmp_path / ".env").write_text('export NEBIUS_API_KEY="dummy-review-key" # local comment\nTAVILY_API_KEY=\'dummy-web-key\'\n')
    (tmp_path / "config.yaml").write_text('llm:\n  providers:\n    nebius:\n      api_key: "${NEBIUS_API_KEY:}"\n')
    cfg = load_config(str(tmp_path / "config.yaml"))
    assert cfg["llm"]["providers"]["nebius"]["api_key"] == "dummy-review-key"
    assert os.environ["TAVILY_API_KEY"] == "dummy-web-key"


def test_private_key_file_can_supply_a_server_secret(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    key = tmp_path / "private-key"
    key.write_text("dummy-review-key\n")
    monkeypatch.setenv("NEBIUS_API_KEY_FILE", str(key))
    (tmp_path / "config.yaml").write_text('llm:\n  providers:\n    nebius:\n      api_key: "${NEBIUS_API_KEY:}"\n')
    cfg = load_config(str(tmp_path / "config.yaml"))
    assert cfg["llm"]["providers"]["nebius"]["api_key"] == "dummy-review-key"
