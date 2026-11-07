from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Asset:
    exchange: str
    ticker: str
    name: str
    keywords: tuple[str, ...]

    @property
    def code(self) -> str:
        return self.ticker.split(".")[0]


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    assets: tuple[Asset, ...]

    def exchange(self, code: str) -> dict[str, Any]:
        return self.raw["exchanges"][code]

    def assets_for(self, exchange: str) -> tuple[Asset, ...]:
        return tuple(a for a in self.assets if a.exchange == exchange)

    def source(self, name: str) -> dict[str, Any]:
        return self.raw["sources"][name]

    @property
    def exchanges(self) -> list[str]:
        return list(self.raw["exchanges"].keys())


def load_config(path: str | Path = ROOT / "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assets = tuple(
        Asset(
            exchange=ex_code,
            ticker=a["ticker"],
            name=a["name"],
            keywords=tuple(a["keywords"]),
        )
        for ex_code, ex in raw["exchanges"].items()
        for a in ex["assets"]
    )
    return Config(raw=raw, assets=assets)


def apify_token() -> str | None:
    return os.environ.get("APIFY_API_TOKEN")


def exa_key() -> str | None:
    return os.environ.get("EXA_API_KEY")
