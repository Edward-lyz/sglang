from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, Tuple, Union

import orjson
from fastapi import HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.content_detection.config import DETECTION_CONFIG
from sglang.srt.entrypoints.openai.encoding_dsv32 import DS32EncodingError
from sglang.srt.entrypoints.openai.protocol import ErrorResponse, OpenAIServingRequest
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.observability.req_time_stats import monotonic_time
from sglang.srt.observability.request_trace import trace, traced_rid
from sglang.srt.runtime_context import get_observability
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


# Base class for specific endpoint handlers
class OpenAIServingBase(ABC):
    """Abstract base class for OpenAI endpoint handlers"""

    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager
        self.allowed_custom_labels = (
            set(get_observability().tokenizer_metrics_allowed_custom_labels)
            if isinstance(self.tokenizer_manager.server_args, ServerArgs)
            and get_observability().tokenizer_metrics_allowed_custom_labels
            else None
        )

    def _parse_model_parameter(self, model: str) -> Tuple[str, Optional[str]]:
        """Parse 'base-model:adapter-name' syntax to extract LoRA adapter.

        Returns (base_model, adapter_name) or (model, None) if no colon present.
        """
        if ":" not in model:
            return model, None

        # Split on first colon only to handle model paths with multiple colons
        parts = model.split(":", 1)
        base_model = parts[0].strip()
        adapter_name = parts[1].strip() or None

        return base_model, adapter_name

    def _resolve_lora_path(
        self,
        request_model: str,
        explicit_lora_path: Optional[Union[str, List[Optional[str]]]],
    ) -> Optional[Union[str, List[Optional[str]]]]:
        """Resolve LoRA adapter with priority: model parameter > explicit lora_path.

        Returns adapter name or None. Supports both single values and lists (batches).
        """
        _, adapter_from_model = self._parse_model_parameter(request_model)

        # Model parameter adapter takes precedence
        if adapter_from_model is not None:
            return adapter_from_model

        # Fall back to explicit lora_path
        return explicit_lora_path

    async def handle_request(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Union[Any, StreamingResponse, ErrorResponse]:
        """Handle the specific request type with common pattern
        If you want to override this method, you should be careful to record the validation time.
        """
        received_time = monotonic_time()
        if rid := traced_rid(getattr(request, "rid", None)):
            trace(logger, "api server received request %s", rid)

        try:
            # Validate request
            error_msg = self._validate_request(request)
            if error_msg:
                return self.create_error_response(error_msg)

            # Log the raw OpenAI request payload before conversion to tokenized form.
            request_logger = self.tokenizer_manager.request_logger
            if request_logger.log_requests and request_logger.log_requests_level >= 2:
                request_logger.log_openai_received_request(request, request=raw_request)

            # Convert to internal format
            adapted_request, processed_request = self._convert_to_internal_request(
                request, raw_request
            )

            if isinstance(adapted_request, (GenerateReqInput, EmbeddingReqInput)):
                # Only set timing fields if adapted_request supports them
                adapted_request.received_time = received_time

            self._process_trace_config(request, adapted_request)

            # Note(Xinyuan): raw_request below is only used for detecting the connection of the client
            if hasattr(request, "stream") and request.stream:
                return await self._handle_streaming_request(
                    adapted_request, processed_request, raw_request
                )
            else:
                return await self._handle_non_streaming_request(
                    adapted_request, processed_request, raw_request
                )
        except HTTPException as e:
            return self.create_error_response(
                message=e.detail, err_type=str(e.status_code), status_code=e.status_code
            )
        except ValueError as e:
            return self.create_error_response(
                message=str(e),
                err_type="BadRequest",
                status_code=400,
            )
        except DS32EncodingError as e:
            logger.info(f"DS32EncodingError: {e}")
            return self.create_error_response(
                message=str(e),
                err_type="BadRequest",
                status_code=400,
            )
        except Exception as e:
            logger.exception(f"Error in request: {e}")
            return self.create_error_response(
                message=f"Internal server error: {str(e)}",
                err_type="InternalServerError",
                status_code=500,
            )

    @abstractmethod
    def _request_id_prefix(self) -> str:
        """Generate request ID based on request type"""
        pass

    def _generate_request_id_base(self, request: OpenAIServingRequest) -> Optional[str]:
        """Generate request ID based on request type"""
        return None

        # TODO(chang): the rid is used in io_strcut check and often violates `The rid should be a list` AssertionError
        # Temporarily return None in this function until the rid logic is clear.
        if rid := getattr(request, "rid", None):
            return rid

        return f"{self._request_id_prefix()}{uuid.uuid4().hex}"

    @abstractmethod
    def _convert_to_internal_request(
        self,
        request: OpenAIServingRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, OpenAIServingRequest]:
        """Convert OpenAI request to internal format"""
        pass

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse, ORJSONResponse]:
        """Handle streaming request

        Override this method in child classes that support streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> Union[Any, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming request

        Override this method in child classes that support non-streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support non-streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    def _validate_request(self, _: OpenAIServingRequest) -> Optional[str]:
        """Validate request"""
        pass

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
        param: Optional[str] = None,
    ) -> ORJSONResponse:
        """Create an error response"""
        # TODO: remove fastapi dependency in openai and move response handling to the entrypoint
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=param,
            code=status_code,
        )
        return ORJSONResponse(content=error.model_dump(), status_code=status_code)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        """Create a streaming error response"""
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=None,
            code=status_code,
        )
        return json.dumps({"error": error.model_dump()})

    @staticmethod
    def raise_abort_before_stream(finish_reason: dict) -> NoReturn:
        """Re-raise a scheduler abort that arrived before any SSE chunk was sent.

        A StreamingResponse commits HTTP 200 the moment it is returned, so an
        abort can only be reported in-band once streaming has started. Before
        that -- while the caller is still inside the kick-start ``__anext__()``
        of ``_handle_streaming_request`` -- raising instead lets
        ``handle_request`` emit the real HTTP status code, matching the
        non-streaming path. The exception mapping mirrors
        ``TokenizerManager._handle_abort_finish_reason`` so both modes produce
        identical response bodies.
        """
        message = finish_reason.get("message", "Generation aborted.")
        code = finish_reason["status_code"]
        if code == HTTPStatus.BAD_REQUEST:
            raise ValueError(message)
        raise HTTPException(status_code=int(code), detail=message)

    def extract_custom_labels(self, raw_request):
        if (
            not self.allowed_custom_labels
            or not get_observability().tokenizer_metrics_custom_labels_header
        ):
            return None

        custom_labels = None
        header = get_observability().tokenizer_metrics_custom_labels_header
        try:
            raw_labels = (
                orjson.loads(raw_request.headers.get(header))
                if raw_request and raw_request.headers.get(header)
                else None
            )
        except json.JSONDecodeError as e:
            logger.exception(f"Error in request: {e}")
            raw_labels = None

        if isinstance(raw_labels, dict):
            custom_labels = {
                label: value
                for label, value in raw_labels.items()
                if label in self.allowed_custom_labels
            }
        return custom_labels

    def extract_routing_key(self, raw_request):
        if raw_request is None:
            return None
        return raw_request.headers.get("x-smg-routing-key")

    def extract_routed_dp_rank_from_header(
        self, raw_request: Request, body_routed_dp_rank: Optional[int] = None
    ) -> Optional[int]:
        """Extract routed_dp_rank from HTTP header, with higher priority than routed_dp_rank in body.

        Header name: X-Data-Parallel-Rank (case-insensitive in HTTP/1.1/2)
        """
        if raw_request is None:
            return body_routed_dp_rank

        header_value = raw_request.headers.get("x-data-parallel-rank")
        if header_value is not None:
            try:
                header_dp_rank = int(header_value)
                if (
                    body_routed_dp_rank is not None
                    and header_dp_rank != body_routed_dp_rank
                ):
                    logger.debug(
                        f"X-Data-Parallel-Rank header ({header_dp_rank}) overrides "
                        f"body routed_dp_rank ({body_routed_dp_rank})"
                    )
                return header_dp_rank
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid X-Data-Parallel-Rank header: must be an integer, got '{header_value}'",
                )

        return body_routed_dp_rank

    @staticmethod
    def _serialize_trace_value(value: Any) -> Any:
        """JSON-ready copy of a request field for the trace payload."""
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [OpenAIServingBase._serialize_trace_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: OpenAIServingBase._serialize_trace_value(item)
                for key, item in value.items()
            }
        return value

    def _build_trace_info(
        self,
        request: OpenAIServingRequest,
        meta_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[dict]:
        """Trace payload the scheduler attached to this request, if any.

        The scheduler-side dump tool that fills meta_info["trace_info"] is not
        part of this delivery, so this returns None until it lands; the response
        field exists so the gateway contract does not change when it does.
        """
        if not DETECTION_CONFIG.enable_trace or not request.trace_config:
            return None
        if not meta_info or "trace_info" not in meta_info:
            return None
        return dict(meta_info["trace_info"])

    def _process_trace_config(self, request, adapted_request) -> None:
        """Resolve trace settings onto the internal request.

        A trace_config only survives when tracing is enabled, and asking for a
        logprobs trace has to turn logprobs on, otherwise the trace would be
        assembled from data the scheduler was never told to produce.
        """
        trace_config = getattr(request, "trace_config", None)
        if not trace_config:
            return
        if not DETECTION_CONFIG.enable_trace:
            logger.info(
                "CD_ENABLE_TRACE is not set, so the trace config of req %s is ignored",
                getattr(request, "rid", None),
            )
            trace_config["enable_trace"] = False

        trace_type = trace_config.get("trace_type")
        if trace_type in ("prompt", "all"):
            if hasattr(request, "messages"):
                trace_config["payload"] = {
                    "messages": self._serialize_trace_value(request.messages)
                }
            elif hasattr(request, "prompt"):
                trace_config["payload"] = {
                    "prompt": self._serialize_trace_value(request.prompt)
                }
        if trace_type in ("logprobs", "all"):
            adapted_request.return_logprob = True
            if "top_logprobs" in trace_config:
                adapted_request.top_logprobs_num = trace_config.get("top_logprobs")

        adapted_request.trace_config = trace_config
