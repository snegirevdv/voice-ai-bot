from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message
from loguru import logger

from app.file_manager import FileManager
from app.lexicon import MessageText
from app.openai import OpenAIClient, OpenAIRateLimitError, OpenAITimeoutError


class VoiceAIBot:
    def __init__(self, bot_token: str, temp_dir: str, client: OpenAIClient):
        self.bot = Bot(token=bot_token)
        self.dp = Dispatcher()
        self.client = client
        self.file_manager = FileManager(temp_dir)
        self.user_threads: dict[int, str] = {}
        logger.info('Инициализация бота завершена')

    def setup_handlers(self):
        self.dp.message.register(self.handle_start, Command('start', 'help'))
        self.dp.message.register(self.handle_voice, F.voice)
        self.dp.message.register(self.handle_text, F.text)
        logger.info('Обработчики сообщений зарегистрированы')

    async def start_polling(self):
        try:
            self.setup_handlers()
            logger.info('Запуск бота...')
            await self.dp.start_polling(self.bot)

        except Exception as e:
            error_msg = f'Ошибка при запуске бота: {e!s}'
            logger.critical(error_msg)
            raise RuntimeError(error_msg) from e

    async def handle_start(self, message: Message):
        """Отправляет приветственного сообщения и инструкции."""
        user_id = message.from_user.id
        logger.info(f'Пользователь {user_id} запустил бота командой /start или /help')
        await message.answer(MessageText.start)

    async def handle_voice(self, message: Message):
        """Обрабатывает голосовое сообщение и отправляет ответное голосове сообщение."""
        user_id = message.from_user.id
        logger.info(f'Получено голосовое сообщение от пользователя {user_id}')

        msg = await message.answer(MessageText.processing_voice)

        voice_file_path = None
        answer_file_path = None

        try:
            voice_file_path = await self._download_voice_message(message)
            logger.debug(f'Голосовое сообщение сохранено: {voice_file_path}')

            transcribed_text = await self._transcribe_voice(voice_file_path)
            logger.info('Голосовое сообщение транскрибировано')

            await message.answer(MessageText.heard.format(text=transcribed_text))
            await msg.delete()
            msg_deleted = True

            answer_file_path = await self._process_and_respond(
                message,
                transcribed_text,
            )

        except OpenAIRateLimitError as e:
            await self._handle_rate_limit_error(message, e)

        except OpenAITimeoutError as e:
            await self._handle_timeout_error(message, e)

        except Exception as e:
            await self._handle_generic_error(message, e)

        finally:
            if not msg_deleted:
                await msg.delete()

            if voice_file_path:
                await self.file_manager.delete_file(voice_file_path)

            if answer_file_path:
                await self.file_manager.delete_file(answer_file_path)

    async def handle_text(self, message: Message):
        """Обрабатывает текстовое сообщение и отвечает голосом."""
        user_id = message.from_user.id
        logger.info(f'Получено текстовое сообщение от пользователя {user_id}')
        answer_file_path = None

        msg = await message.answer(MessageText.processing_text)

        try:
            answer_file_path = await self._process_and_respond(message, message.text)

        except OpenAIRateLimitError as e:
            await self._handle_rate_limit_error(message, e)

        except OpenAITimeoutError as e:
            await self._handle_timeout_error(message, e)

        except Exception as e:
            await self._handle_generic_error(message, e)

        finally:
            await msg.delete()

            if answer_file_path:
                await self.file_manager.delete_file(answer_file_path)

    async def _download_voice_message(self, message: Message) -> Path:
        """Загружает голосовое сообщение. Возвращает путь к сохраненному голосовому файлу."""
        user_id = message.from_user.id
        file_id = message.voice.file_id

        voice_file_path = self.file_manager.get_voice_path(user_id, file_id)
        voice_file = await self.bot.get_file(file_id)

        await self.bot.download_file(voice_file.file_path, voice_file_path)
        logger.debug(f'Голосовое сообщение загружено в {voice_file_path}')

        return voice_file_path

    async def _transcribe_voice(self, voice_file_path: Path) -> str:
        """Преобразует голосовой файл в текст."""
        with voice_file_path.open('rb') as file:
            return await self.client.speech_to_text(file)

    async def _process_and_respond(self, message: Message, text: str) -> Path:
        """Обрабатывает входящий текст и отправляет голосовой ответ."""
        user_id = message.from_user.id

        response_text, _ = await self._get_response(user_id, text)
        logger.info(f'Получен ответ от OpenAI для пользователя {user_id}')

        voice_path = await self._create_and_send_voice(message, response_text)
        logger.info(f'Отправлен голосовой ответ пользователю {user_id}')

        return voice_path

    async def _get_response(self, user_id: int, text: str) -> tuple[str, str]:
        """Получает ответ от OpenAI. Возвращает кортеж из текста ответа и ID диалога."""
        thread_id = self.user_threads.get(user_id)
        response_text, thread_id = await self.client.respond(text, thread_id)
        self.user_threads[user_id] = thread_id
        return response_text, thread_id

    async def _create_and_send_voice(self, message: Message, text: str) -> Path:
        """Создает и отправляет голосовое сообщение на основе текста. Возвращает путь к файлу."""
        voice_file_path = await self.client.text_to_speech(text)
        voice_path = Path(voice_file_path)

        await message.answer_voice(voice=FSInputFile(voice_path))
        return voice_path

    async def _handle_rate_limit_error(
        self,
        message: Message,
        error: Exception,
    ) -> None:
        """Обработка ошибки превышения лимита запросов."""
        logger.warning(f'Превышен лимит запросов: {error!s}')
        error_text = 'Превышен лимит запросов к API. Пожалуйста, попробуйте ещё раз.'
        await message.answer(MessageText.error.format(error=error_text))

    async def _handle_timeout_error(self, message: Message, error: Exception) -> None:
        """Обработка ошибки тайм-аута."""
        logger.warning(f'Превышено время ожидания: {error!s}')
        error_text = 'Превышено время ожидания ответа. Пожалуйста, попробуйте ещё раз.'

        await message.answer(MessageText.error.format(error=error_text))

    async def _handle_generic_error(self, message: Message, error: Exception) -> None:
        """Обработка общих ошибок."""
        error_msg = f'Ошибка при обработке сообщения: {error!s}'
        logger.error(error_msg)
        error_text = (
            'Произошла ошибка при обработке сообщения. Пожалуйста, попробуйте ещё раз.'
        )

        await message.answer(MessageText.error.format(error=error_text))
