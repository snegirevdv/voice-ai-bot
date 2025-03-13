import asyncio
import sys

from loguru import logger

from app.bot import VoiceAIBot
from app.config import settings
from app.openai import OpenAIClient


def setup_logger():
    """Настройка логирования."""
    logger.remove()

    logger.add(
        sys.stdout,
        format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level: <8} | <yellow>{name: <25}</yellow> | <cyan>{line: <3}</cyan> | {message}',
        level='DEBUG',
    )

    logger.info('Логирование настроено')


async def periodic_cleanup(bot: VoiceAIBot, sleep: int = 3600):
    """Периодическая очистка временных файлов."""
    while True:
        try:
            await bot.file_manager.cleanup_old_files()
        except Exception as e:
            logger.error(f'Ошибка при периодической очистке файлов: {e!s}')

        await asyncio.sleep(sleep)


async def main():
    """Основная функция для запуска бота."""
    setup_logger()

    openai_client = OpenAIClient(
        token=settings.OPENAI_TOKEN,
        temp_dir=settings.TEMP_DIR,
        instruction=settings.INSTRUCTION,
    )
    await openai_client.initialize_assistant()

    bot = VoiceAIBot(
        bot_token=settings.BOT_TOKEN,
        temp_dir=settings.TEMP_DIR,
        client=openai_client,
    )

    cleanup_task = asyncio.create_task(periodic_cleanup(bot))

    try:
        await bot.start_polling()
        logger.info('Бот запущен')

    except Exception as e:
        logger.critical(f'Ошибка при работе бота: {e!s}')

    finally:
        cleanup_task.cancel()
        logger.info('Бот остановлен')


if __name__ == '__main__':
    asyncio.run(main())
