import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from loguru import logger
from openai import AsyncOpenAI, HttpxBinaryResponseContent


class OpenAIServiceError(Exception):
    """Базовое исключение для ошибок OpenAI."""


class OpenAIRateLimitError(OpenAIServiceError):
    """Исключение для ошибок превышения лимита запросов."""


class OpenAITimeoutError(OpenAIServiceError):
    """Исключение для ошибок тайм-аута."""


class OpenAIClient:
    """Клиент для взаимодействия с API OpenAI."""

    def __init__(self, token: str, temp_dir: str, instruction: str):
        self.token = token
        self.client = AsyncOpenAI(api_key=self.token)
        self.assistant_id = None
        self.instruction = instruction

        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f'Временная директория создана: {self.temp_dir}')

    async def initialize_assistant(self):
        """Получает существующий или создает новый ID ассистента."""
        assistants = await self.client.beta.assistants.list(limit=1)

        if assistants.data and assistants.data[0].name == 'tg-ai-voice-bot':
            assistant = assistants.data[0]
            logger.debug(f'Найден существующий ассистент с ID: {assistant.id}')

        else:
            assistant = await self.client.beta.assistants.create(
                name='tg-ai-voice-bot',
                instructions=self.instruction,
                model='gpt-4o',
            )

            logger.debug(f'Создан новый ассистент с ID: {assistant.id}')

        self.assistant_id = assistant.id
        logger.info(f'Установлен Assistant ID = {self.assistant_id}')

    async def respond(
        self,
        user_message: str,
        thread_id: str | None = None,
    ) -> tuple[str, str]:
        """
        Получение ответа от ассистента OpenAI на сообщение пользователя.
        Возвращает кортеж из текста ответа и ID диалога.
        """
        try:
            if not thread_id:
                thread = await self.client.beta.threads.create()
                thread_id = thread.id
                logger.info(f'Создан новый диалог с ID: {thread_id}')
            else:
                logger.debug(f'Используется существующий диалог с ID: {thread_id}')

            await self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role='user',
                content=user_message,
            )
            logger.debug(f'Сообщение пользователя добавлено в диалог {thread_id}')

            run = await self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id,
            )

            await self._wait_for_run_completion(thread_id, run.id)
            logger.debug('Диалог обработан, извлекаем ответ')

            response, thread_id = await self._extract_assistant_response(thread_id)
            logger.info('Получен ответ от ассистента')

            return response, thread_id

        except asyncio.TimeoutError as e:
            msg = f'Превышено время ожидания ответа от OpenAI: {e!s}'
            logger.error(msg)
            raise OpenAITimeoutError(msg) from e

        except Exception as e:
            if 'rate_limit' in str(e).lower() or 'too many requests' in str(e).lower():
                msg = f'Превышен лимит запросов к API OpenAI: {e!s}'
                logger.error(msg)
                raise OpenAIRateLimitError(msg) from e

            msg = f'Ошибка при получении ответа от ассистента: {e!s}'
            logger.error(msg)
            raise OpenAIServiceError(msg) from e

    async def speech_to_text(self, audio_file: BinaryIO) -> str:
        """Преобразует аудио файл в текст."""
        try:
            response = await self.client.audio.transcriptions.create(
                file=audio_file,
                model='whisper-1',
            )
            text = response.text
            logger.info('Аудио успешно транскрибировано')
            return text

        except Exception as e:
            error_msg = f'Ошибка при транскрибировании аудио: {e!s}'
            logger.error(error_msg)
            raise OpenAIServiceError(error_msg) from e

    async def text_to_speech(self, text: str) -> str:
        """Преобразует текст в речь. Возвращает путь к аудио-файлу"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filepath = self.temp_dir / f'response_{timestamp}.mp3'

            response: HttpxBinaryResponseContent = (
                await self.client.audio.speech.create(
                    model='tts-1',
                    voice='echo',
                    speed=1,
                    input=text,
                )
            )

            audio = await response.aread()

            with filepath.open('wb') as file:
                file.write(audio)

            logger.info(f'Текст успешно преобразован в речь, путь к файлу: {filepath}')
            return str(filepath)

        except Exception as e:
            error_msg = f'Ошибка при преобразовании текста в речь: {e!s}'
            logger.error(error_msg)
            raise OpenAIServiceError(error_msg) from e

    async def _wait_for_run_completion(
        self,
        thread_id: str,
        run_id: str,
        timeout: int = 60,
    ) -> None:
        """Ожидает завершение выполнения запроса ассистента."""
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                msg = f'Превышено время ожидания ответа ({timeout} сек)'
                raise OpenAITimeoutError(msg)

            run_status = await self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id,
            )

            if run_status.status == 'completed':
                logger.debug('Запрос к ассистенту выполнен успешно')
                return

            if run_status.status in ['failed', 'cancelled', 'expired']:
                msg = f'Обработка запроса не выполнена, статус: {run_status.status}'
                logger.error(msg)
                raise OpenAIServiceError(msg)

            await asyncio.sleep(1)

    async def _extract_assistant_response(self, thread_id: str) -> tuple[str, str]:
        """
        Извлекает ответ ассистента из сообщений диалога.
        Возвращает кортеж из текста ответа и ID диалога.
        """
        messages = await self.client.beta.threads.messages.list(thread_id=thread_id)

        for message in messages.data:
            if message.role == 'assistant':
                content_texts = [
                    content.text.value
                    for content in message.content
                    if hasattr(content, 'text')
                ]

                response_text = ' '.join(content_texts)
                logger.debug(
                    f'Извлечен ответ ассистента (символов: {len(response_text)})'
                )
                return (response_text, thread_id)

        response_text = 'Извините, в данный момент у меня нет ответа.'
        logger.warning('Ответ ассистента не найден, возвращаю стандартный ответ')
        return (response_text, thread_id)
