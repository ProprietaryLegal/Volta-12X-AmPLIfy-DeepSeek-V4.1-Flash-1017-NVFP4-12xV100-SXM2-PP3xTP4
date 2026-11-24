# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicTokenizeResponse,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def messages(request: Request) -> AnthropicServingMessages:
    return request.app.state.anthropic_serving_messages


def translate_error_response(response: ErrorResponse) -> JSONResponse:
    anthropic_error = AnthropicErrorResponse(
        error=AnthropicError(
            type=response.error.type,
            message=response.error.message,
        )
    )
    return JSONResponse(
        status_code=response.error.code, content=anthropic_error.model_dump()
    )


@router.post(
    "/v1/messages",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": AnthropicErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": AnthropicErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": AnthropicErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_messages(request: AnthropicMessagesRequest, raw_request: Request):
    handler = messages(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        error = base_server.create_error_response(
            NotImplementedError("The model does not support Messages API")
        )
        return translate_error_response(error)

    try:
        generator = await handler.create_messages(request, raw_request)
    except ValueError as e:
        # Same mapping as entrypoints.utils.create_error_response: a ValueError
        # means this request cannot be accepted, so retrying it cannot succeed.
        logger.warning("Rejected create_messages request: %s", e)
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST.value,
            content=AnthropicErrorResponse(
                error=AnthropicError(
                    type="invalid_request_error",
                    message=str(e),
                )
            ).model_dump(),
        )
    except Exception as e:
        logger.exception("Error in create_messages: %s", e)
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content=AnthropicErrorResponse(
                error=AnthropicError(
                    type="internal_error",
                    message=str(e),
                )
            ).model_dump(),
        )

    if isinstance(generator, ErrorResponse):
        return translate_error_response(generator)

    elif isinstance(generator, AnthropicMessagesResponse):
        resp = generator.model_dump(exclude_none=True)
        logger.debug("Anthropic Messages Response: %s", resp)
        return JSONResponse(content=resp)

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/messages/count_tokens",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": AnthropicCountTokensResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": AnthropicErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": AnthropicErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": AnthropicErrorResponse},
    },
)
@load_aware_call
@with_cancellation
async def count_tokens(request: AnthropicCountTokensRequest, raw_request: Request):
    handler = messages(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        error = base_server.create_error_response(
            NotImplementedError("The model does not support Messages API")
        )
        return translate_error_response(error)

    try:
        response = await handler.count_tokens(request, raw_request)
    except ValueError as e:
        # Same mapping as entrypoints.utils.create_error_response: a ValueError
        # means this request cannot be accepted, so retrying it cannot succeed.
        logger.warning("Rejected count_tokens request: %s", e)
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST.value,
            content=AnthropicErrorResponse(
                error=AnthropicError(
                    type="invalid_request_error",
                    message=str(e),
                )
            ).model_dump(),
        )
    except Exception as e:
        logger.exception("Error in count_tokens: %s", e)
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content=AnthropicErrorResponse(
                error=AnthropicError(
                    type="internal_error",
                    message=str(e),
                )
            ).model_dump(),
        )

    if isinstance(response, ErrorResponse):
        return translate_error_response(response)

    return JSONResponse(content=response.model_dump(exclude_none=True))


@router.post(
    "/v1/messages/tokenize",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": AnthropicTokenizeResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": AnthropicErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": AnthropicErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": AnthropicErrorResponse},
    },
)
@with_cancellation
async def tokenize_messages(request: AnthropicMessagesRequest, raw_request: Request):
    """The prompt token ids /v1/messages would run for the same body (item 5 of
    the store work: a client warms a document pack from the body its chats send).
    Not load-tracked: nothing is generated."""
    handler = messages(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        error = base_server.create_error_response(
            NotImplementedError("The model does not support Messages API")
        )
        return translate_error_response(error)

    try:
        response = await handler.tokenize_messages(request)
    except ValueError as e:
        # As /v1/messages: a ValueError means this body cannot be accepted.
        logger.warning("Rejected messages tokenize request: %s", e)
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST.value,
            content=AnthropicErrorResponse(
                error=AnthropicError(
                    type="invalid_request_error",
                    message=str(e),
                )
            ).model_dump(),
        )
    except Exception as e:
        logger.exception("Error in messages tokenize: %s", e)
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content=AnthropicErrorResponse(
                error=AnthropicError(
                    type="internal_error",
                    message=str(e),
                )
            ).model_dump(),
        )

    if isinstance(response, ErrorResponse):
        return translate_error_response(response)

    return JSONResponse(content=response.model_dump())


def attach_router(app: FastAPI):
    app.include_router(router)
