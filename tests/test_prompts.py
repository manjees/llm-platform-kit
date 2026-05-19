"""Unit tests for llm_kit.prompts — fail-fast behavior is critical."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_kit.prompts import (
    BrainFileMissing,
    BrainNotConfigured,
    brain_dir,
    list_files,
    read_text,
    read_yaml,
)


def test_brain_dir_unset_raises(monkeypatch):
    monkeypatch.delenv("BRAIN_DIR", raising=False)
    with pytest.raises(BrainNotConfigured):
        brain_dir()


def test_brain_dir_nonexistent_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAIN_DIR", str(tmp_path / "does_not_exist"))
    with pytest.raises(BrainNotConfigured):
        brain_dir()


def test_brain_dir_file_raises(monkeypatch, tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hi")
    monkeypatch.setenv("BRAIN_DIR", str(f))
    with pytest.raises(BrainNotConfigured):
        brain_dir()


def test_read_text_ok(monkeypatch, tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "hello.md").write_text("hello world", encoding="utf-8")
    monkeypatch.setenv("BRAIN_DIR", str(tmp_path))
    assert read_text("prompts", "hello.md") == "hello world"


def test_read_text_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAIN_DIR", str(tmp_path))
    with pytest.raises(BrainFileMissing):
        read_text("prompts", "missing.md")


def test_read_yaml_ok(monkeypatch, tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "rules.yaml").write_text(
        "name: test\nvalue: 42", encoding="utf-8"
    )
    monkeypatch.setenv("BRAIN_DIR", str(tmp_path))
    data = read_yaml("config", "rules.yaml")
    assert data == {"name": "test", "value": 42}


def test_read_yaml_non_mapping_raises(monkeypatch, tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "list.yaml").write_text("- a\n- b", encoding="utf-8")
    monkeypatch.setenv("BRAIN_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        read_yaml("config", "list.yaml")


def test_list_files(monkeypatch, tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "a.md").write_text("a")
    (tmp_path / "prompts" / "b.md").write_text("b")
    (tmp_path / "prompts" / "ignore.txt").write_text("c")
    monkeypatch.setenv("BRAIN_DIR", str(tmp_path))
    files = list_files("prompts", pattern="*.md")
    assert sorted(f.name for f in files) == ["a.md", "b.md"]
