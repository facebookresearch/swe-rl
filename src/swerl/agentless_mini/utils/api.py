# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import asyncio
import time
from typing import Any, Awaitable, Literal, TypeVar

import openai
import tenacity
from openai.types.chat import ChatCompletion
from tqdm.auto import tqdm

from .envs import ANSWER_END_TAG, ANSWER_START_TAG, THINKING


def retry(errors: Any, max_attempts: int = 5):
    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(errors),
        wait=tenacity.wait_exponential(multiplier=1, min=5, max=20),
        stop=tenacity.stop_after_attempt(max_attempts),
        before_sleep=print,
    )


ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIClient:
    def __init__(self):
        self.client = openai.OpenAI()
        self.async_client = openai.AsyncClient()

    @retry(ERRORS)
    def chat_completions_with_backoff(self, *args, **kwargs):
        return self.client.chat.completions.create(*args, **kwargs)

    @retry(ERRORS)
    def completions_with_backoff(self, *args, **kwargs):
        return self.client.completions.create(*args, **kwargs)

    @retry(ERRORS)
    async def chat_completions_with_backoff_async(self, *args, **kwargs):
        return await self.async_client.chat.completions.create(*args, **kwargs)

    @retry(ERRORS)
    async def completions_with_backoff_async(self, *args, **kwargs):
        return await self.async_client.completions.create(*args, **kwargs)

    async def safe_chat_completion(self, request: dict):
        try:
            return await self.chat_completions_with_backoff_async(**request)
        except openai.BadRequestError as e:
            print("Error request:", str(e))
            return None

    async def delayed_request(
        self,
        request: dict[str, Any],
        mode: Literal["chat", "completion"],
        delay: float | None,
    ):
        """Prevent quantized rate limit:
        https://help.openai.com/en/articles/6891753-rate-limit-advice"""
        if delay is not None:
            # synchronized sleep
            time.sleep(delay)
        if mode == "chat":
            func = self.chat_completions_with_backoff_async
        else:
            func = self.completions_with_backoff_async
        return await func(**request)

    def dispatch_chat_completions(
        self,
        requests: list[dict[str, Any]],
        delay: float | None = None,
    ):
        return asyncio.run(self._dispatch_chat_completions(requests, delay))

    def dispatch_completions(
        self,
        requests: list[dict[str, Any]],
        delay: float | None = None,
    ):
        return asyncio.run(self._dispatch_completions(requests, delay))

    async def _dispatch_chat_completions(
        self,
        requests: list[dict[str, Any]],
        delay: float | None = None,
    ):
        """Dispatch chat completions requests asynchronously.
        Args:
            requests: a list of API argument names to values.
            delay: interval between requests.
        """

        tasks = [self.delayed_request(request, "chat", delay) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _dispatch_completions(
        self,
        requests: list[dict[str, Any]],
        delay: float | None = None,
    ):
        """Dispatch completions requests asynchronously.
        Args:
            requests: a list of API argument names to values.
            delay: interval between requests.
        """

        tasks = [
            self.delayed_request(request, "completion", delay) for request in requests
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)


T = TypeVar("T")


async def run_with_semaphore(
    semaphore: asyncio.Semaphore, task: Awaitable[T], index: int
):
    async with semaphore:
        return (index, await task)


async def collect_responses_async(
    client: OpenAIClient,
    semaphore: asyncio.Semaphore,
    all_requests: list[dict],
):
    all_tasks = [
        run_with_semaphore(semaphore, client.safe_chat_completion(request), idx)
        for idx, request in enumerate(all_requests)
    ]
    idx_and_responses = list[tuple[int, ChatCompletion | None]]()
    pbar = tqdm(total=len(all_tasks), desc="Process each instance", leave=False)
    for completion in asyncio.as_completed(all_tasks):
        idx, response = await completion
        idx_and_responses.append((idx, response))
        pbar.update(1)
    pbar.close()
    return idx_and_responses


def parse_thinking_output(output: str) -> str:
    """Extract the <solution> part for thinking models"""
    if THINKING:
        output = output.split(ANSWER_START_TAG, 1)[-1]
        output = output.split(ANSWER_END_TAG, 1)[0]
    return output.strip()
