import time
from pathlib import Path

from loguru import logger


class FileManager:
    """Менеджер для работы с файлами."""

    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self._ensure_temp_dir()
        logger.info(f'Менеджер файлов инициализирован с директорией: {self.temp_dir}')

    def _ensure_temp_dir(self) -> None:
        """Создает временную директорию, если она не существует."""
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f'Проверена/создана временная директория: {self.temp_dir}')

    def get_voice_path(self, user_id: int, file_id: str) -> Path:
        """Возвращает путь для сохранения голосового сообщения пользователя."""
        return self.temp_dir / f'voice_{user_id}_{file_id}.ogg'

    async def delete_file(self, file_path: Path) -> None:
        """Удаляет файл."""
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f'Удален файл: {file_path}')
            else:
                logger.debug(f'Файл не найден для удаления: {file_path}')

        except Exception as e:
            logger.warning(f'Не удалось удалить файл {file_path}: {e!s}')

    async def cleanup_old_files(self, max_age_seconds: int = 600) -> int:
        """Удаляет старые временные файлы."""
        deleted_count = 0
        current_time = time.time()

        try:
            for file_path in self.temp_dir.glob('*'):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime

                    if file_age > max_age_seconds:
                        await self.delete_file(file_path)
                        deleted_count += 1

            msg = f'Очистка старых файлов завершена, удалено {deleted_count} файлов'
            logger.info(msg)
            return deleted_count

        except Exception as e:
            logger.error(f'Ошибка при очистке старых файлов: {e!s}')
            return deleted_count
