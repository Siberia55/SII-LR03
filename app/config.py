from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "keywords.yaml"


def load_config() -> dict:
    """Загружает конфигурацию сценария из YAML."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Конфигурация должна быть YAML-объектом")

    required_fields = {
        "scenario_id",
        "scenario_name",
        "keywords",
    }

    missing_fields = required_fields - config.keys()
    if missing_fields:
        raise ValueError(
            f"В конфигурации отсутствуют поля: {sorted(missing_fields)}"
        )

    if not isinstance(config["keywords"], list) or not config["keywords"]:
        raise ValueError("Список keywords должен быть непустым")

    config["keywords"] = [
        str(keyword).strip().lower()
        for keyword in config["keywords"]
        if str(keyword).strip()
    ]

    return config
