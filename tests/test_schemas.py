import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest
from app.schemas.extract import AfterSalesTicket, ExtractRequest, RequestType


def test_chat_request_rejects_empty_or_whitespace_fields():
    for payload in (
        {"session_id": "s1", "message": ""},
        {"session_id": "s1", "message": " \t\n "},
        {"session_id": " \t", "message": "hello"},
    ):
        with pytest.raises(ValidationError):
            ChatRequest(**payload)


def test_chat_request_accepts_boundary_lengths():
    assert ChatRequest(session_id="s" * 128, message="m" * 20_000)


def test_chat_request_rejects_values_over_maximum_length():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s" * 129, message="hello")
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="m" * 20_001)


def test_extract_request_rejects_empty_or_whitespace_text():
    for text in ("", " \t\n "):
        with pytest.raises(ValidationError):
            ExtractRequest(text=text)


def test_extract_request_accepts_boundary_length_and_rejects_overflow():
    assert ExtractRequest(text="x" * 20_000)
    with pytest.raises(ValidationError):
        ExtractRequest(text="x" * 20_001)


def test_ticket_order_id_is_nullable_and_defaults_to_none():
    t = AfterSalesTicket(request_type="退款", expected_solution="退全款")
    assert t.order_id is None
    assert t.request_type is RequestType.REFUND


def test_ticket_rejects_unknown_request_type_and_blank_solution():
    with pytest.raises(ValidationError):
        AfterSalesTicket(order_id=None, request_type="砍价", expected_solution="x")
    with pytest.raises(ValidationError):
        AfterSalesTicket(order_id=None, request_type="退款", expected_solution="  ")


def test_request_type_has_only_the_five_supported_values():
    assert [item.value for item in RequestType] == ["退款", "换货", "维修", "投诉", "其他"]
