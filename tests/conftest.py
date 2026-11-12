"""pytest 公共配置：把仓库根目录加入 sys.path，并提供最小化的 session_state 桩。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeSessionState(dict):
    """支持 st.session_state.xxx 与 st.session_state["xxx"] 两种写法。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


@pytest.fixture
def fake_state(monkeypatch):
    import streamlit as st

    state = FakeSessionState()
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(st, "secrets", {})
    return state
