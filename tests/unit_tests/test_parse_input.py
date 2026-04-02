"""Tests for parse_input node -- multipart message handling."""

from langchain_core.messages import HumanMessage

from agent.graph import State, parse_input


def _make_state(content):
    """Build a minimal State with one HumanMessage."""
    return State(messages=[HumanMessage(content=content)])


def test_multipart_with_pdf():
    """Multipart content with .pdf filename extracts question and source_file."""
    state = _make_state([
        {"type": "text", "text": "What are the compliance requirements?"},
        {"type": "text", "text": "06MC010425F12568B380B14D9CBFEC5270EA9F5FF3.pdf"},
    ])
    result = parse_input(state)
    assert result["source_file"] == "06MC010425F12568B380B14D9CBFEC5270EA9F5FF3.pdf"
    msg = result["messages"][0]
    assert msg.content == "What are the compliance requirements?"


def test_multipart_pdf_first():
    """Filename can appear before the question text."""
    state = _make_state([
        {"type": "text", "text": "report.pdf"},
        {"type": "text", "text": "Summarize this document"},
    ])
    result = parse_input(state)
    assert result["source_file"] == "report.pdf"
    msg = result["messages"][0]
    assert msg.content == "Summarize this document"


def test_plain_text_message():
    """Plain string content has no source_file, message unchanged."""
    state = _make_state("What are the compliance requirements?")
    result = parse_input(state)
    assert result["source_file"] == ""
    msg = result["messages"][0]
    assert msg.content == "What are the compliance requirements?"


def test_multipart_no_pdf():
    """Multipart content with no .pdf filename has no source_file, text concatenated."""
    state = _make_state([
        {"type": "text", "text": "What are the compliance requirements?"},
        {"type": "text", "text": "for banking sector"},
    ])
    result = parse_input(state)
    assert result["source_file"] == ""
    msg = result["messages"][0]
    assert "compliance requirements" in msg.content
    assert "banking sector" in msg.content
