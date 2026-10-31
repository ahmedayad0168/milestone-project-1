from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")


def test_ui_chat_flow_with_followup(kb):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("Extractive" in c.value or "LLM" in c.value for c in at.sidebar.caption)

    at.chat_input[0].set_value("What confidence level does Expected Shortfall use under FRTB?").run()
    assert not at.exception, at.exception
    roles = [m.name for m in at.chat_message]
    assert roles == ["user", "assistant"]
    assert "97.5%" in at.chat_message[1].markdown[0].value

    at.chat_input[0].set_value("and what about the liquidity horizons?").run()
    assert not at.exception, at.exception
    assert len(at.chat_message) == 4
    assert any("Interpreted as" in c.value for c in at.chat_message[3].caption)
