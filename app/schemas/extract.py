from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ExtractRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        min_length=1,
        max_length=20_000,
        description="用户的售后描述原文",
    )


class RequestType(str, Enum):
    REFUND = "退款"
    EXCHANGE = "换货"
    REPAIR = "维修"
    COMPLAINT = "投诉"
    OTHER = "其他"


class AfterSalesTicket(BaseModel):
    """从用户售后描述中提取的结构化工单。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    order_id: str | None = Field(
        default=None,
        description="订单号,原文未出现则为 null,禁止编造",
    )
    request_type: RequestType = Field(description="用户诉求类型")
    expected_solution: str = Field(
        min_length=1,
        max_length=20_000,
        description="用户期望的处理方案,一句话概括",
    )
