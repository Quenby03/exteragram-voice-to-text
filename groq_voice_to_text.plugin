# ==============================================================================
# Groq Whisper Voice to Text Plugin for exteraGram
# Baseline: SDK >= 1.4.4.3 | Python 3.11 | exteraGram >= 12.5.1
# ==============================================================================

import http.client
import json
import os
import ssl
import uuid
from typing import Any, List, Optional

from base_plugin import BasePlugin, HookResult, HookStrategy
from client_utils import run_on_queue, send_text
from ui.bulletin import BulletinHelper
from ui.settings import Divider, Header, Input, Switch, Text

__id__ = "groq_voice_to_text"
__name__ = "Groq Voice to Text"
__description__ = (
    "Перехватывает голосовые сообщения (ГС), отправляет их в Groq Whisper API "
    "через отдельное кратковременное HTTP-соединение и шлёт распознанный текст "
    "в тот же чат ответом на исходное сообщение."
)
__author__ = "@Quenby03"
__version__ = "1.0.0"
__icon__ = "exteraPlugins/1"
__app_version__ = ">=12.5.1"
__sdk_version__ = ">=1.4.4.3"
__category__ = ["media", "tools"]

DEFAULT_MODEL = "whisper-large-v3-turbo"


def transcribe_groq_whisper(
    audio_bytes: bytes,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 35,
) -> str:
    """
    Отправляет аудио в Groq Whisper API используя единственное HTTP/HTTPS соединение,
    которое существует исключительно во время выполнения запроса и сразу закрывается.
    Формат ответа: json (без таймкодов).
    """
    boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
    crlf = b"\r\n"
    parts: List[bytes] = []

    # Поле 1: model
    parts.append(b"--" + boundary.encode("ascii"))
    parts.append(b'Content-Disposition: form-data; name="model"')
    parts.append(b"")
    parts.append(model.strip().encode("utf-8"))

    # Поле 2: response_format (строго "json" для получения обычного текста без таймкодов)
    parts.append(b"--" + boundary.encode("ascii"))
    parts.append(b'Content-Disposition: form-data; name="response_format"')
    parts.append(b"")
    parts.append(b"json")

    # Поле 3: file (бинарные данные аудиофайла)
    parts.append(b"--" + boundary.encode("ascii"))
    parts.append(b'Content-Disposition: form-data; name="file"; filename="voice.ogg"')
    parts.append(b"Content-Type: audio/ogg")
    parts.append(b"")
    parts.append(audio_bytes)

    # Завершающий boundary
    parts.append(b"--" + boundary.encode("ascii") + b"--")
    parts.append(b"")

    body = crlf.join(parts)

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
        "Connection": "close",
        "User-Agent": "exteraGram-GroqWhisperPlugin/1.0",
    }

    try:
        context = ssl.create_default_context()
    except Exception:
        context = None

    # Одно HTTP-соединение: создаётся -> выполняется запрос -> закрывается
    if context:
        conn = http.client.HTTPSConnection("api.groq.com", 443, context=context, timeout=timeout)
    else:
        conn = http.client.HTTPSConnection("api.groq.com", 443, timeout=timeout)

    try:
        conn.request("POST", "/openai/v1/audio/transcriptions", body=body, headers=headers)
        response = conn.getresponse()
        raw_resp = response.read()
        resp_text = raw_resp.decode("utf-8", errors="replace")

        if response.status != 200:
            error_msg = f"HTTP {response.status}"
            try:
                err_data = json.loads(resp_text)
                if "error" in err_data and "message" in err_data["error"]:
                    error_msg += f": {err_data['error']['message']}"
                else:
                    error_msg += f": {resp_text}"
            except Exception:
                error_msg += f": {resp_text}"
            raise RuntimeError(error_msg)

        data = json.loads(resp_text)
        return data.get("text", "").strip()

    finally:
        # Соединение гарантированно закрывается сразу после использования
        try:
            conn.close()
        except Exception:
            pass


class GroqVoiceToTextPlugin(BasePlugin):
    """
    Плагин для exteraGram:
    - Перехватывает отправку записанных голосовых сообщений (ГС).
    - Блокирует отправку аудиофайла.
    - В фоновом потоке передаёт аудио в Groq Whisper API (response_format=json).
    - Отправляет распознанный текст в тот же чат ответом на то же сообщение (или без ответа).
    """

    def on_plugin_load(self):
        # Регистрируем хук на исходящие сообщения
        self.add_on_send_message_hook()
        self.logger.info("Groq Voice to Text plugin loaded")

    def on_plugin_unload(self):
        self.logger.info("Groq Voice to Text plugin unloaded")

    def create_settings(self) -> List[Any]:
        api_key = self.get_setting("api_key", "").strip()
        if api_key:
            masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
            key_subtext = f"Ключ сохранён ({masked_key})"
        else:
            key_subtext = "Нажмите, чтобы ввести API ключ Groq (gsk_...)"

        current_model = self.get_setting("model", DEFAULT_MODEL).strip() or DEFAULT_MODEL

        return [
            Header(text="Groq Whisper — Голосовые в текст"),
            Switch(
                key="enabled",
                text="Включить замену ГС на текст",
                subtext="Отправлять текст вместо аудиосообщения",
                default=True,
                icon="msg_voice",
            ),
            Input(
                key="api_key",
                text="Groq API Key",
                subtext=key_subtext,
                default="",
                icon="msg_key",
            ),
            Input(
                key="model",
                text="Модель Whisper",
                subtext=f"Текущая модель: {current_model}",
                default=DEFAULT_MODEL,
                icon="msg_settings",
            ),
            Divider(text="Нажмите на пункт, чтобы изменить значение"),
            Text(
                text="О плагине",
                subtext=(
                    "Перехватывает записанные голосовые сообщения, отправляет аудио в "
                    "Groq Whisper API через отдельное соединение и шлёт распознанный текст "
                    "в чат (с сохранением реплая на сообщение, если был ответ)."
                ),
                icon="msg_info",
            ),
        ]

    def on_send_message_hook(self, account: int, params: Any) -> HookResult:
        """
        Перехват исходящего сообщения.
        Если это голосовое сообщение и плагин включен, отменяет отправку аудио
        и запускает транскрибацию через Groq Whisper в фоне.
        """
        # 1. Проверяем, включен ли плагин в настройках
        if not self.get_setting("enabled", True):
            return HookResult()

        # 2. Проверяем, является ли сообщение голосовым
        if not self._is_voice_message(params):
            return HookResult()

        # 3. Проверяем наличие API-ключа Groq
        api_key = self.get_setting("api_key", "").strip()
        if not api_key:
            BulletinHelper.show_error("Groq Whisper: укажите API ключ в настройках плагина!")
            # Не блокируем ГС, если ключ не задан, чтобы пользователь не потерял запись
            return HookResult()

        # 4. Извлекаем параметры чата и реплая
        peer_id = getattr(params, "peer", None)
        if peer_id is None:
            peer_id = getattr(params, "dialog_id", None)

        if peer_id is None:
            self.logger.warning("Could not determine peer_id from params, skipping interception")
            return HookResult()

        reply_to_msg_id = self._extract_reply_to_msg_id(params)
        path = getattr(params, "path", None)

        # 5. Считываем байты аудио сразу в память, чтобы избежать гонки удаления кэша
        audio_bytes = None
        if path and isinstance(path, str):
            try:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    with open(path, "rb") as f:
                        audio_bytes = f.read()
            except Exception as e:
                self.logger.error(f"Failed to read audio file at {path}: {e}")

        # Если файл недоступен и пути нет — пропускаем
        if not audio_bytes and not (path and isinstance(path, str)):
            return HookResult()

        # 6. Получаем выбранную модель
        model = self.get_setting("model", DEFAULT_MODEL).strip() or DEFAULT_MODEL

        # 7. Запускаем фоновую отправку в Groq на очереди плагинов (не блокируя UI-поток)
        run_on_queue(
            lambda: self._process_and_send(
                account=account,
                peer_id=peer_id,
                reply_to_msg_id=reply_to_msg_id,
                audio_bytes=audio_bytes,
                file_path=path,
                api_key=api_key,
                model=model,
            )
        )

        # 8. Отменяем отправку исходного голосового сообщения
        self.logger.info(
            f"Voice message intercepted for peer {peer_id} (replyToMsg: {reply_to_msg_id}). Cancelled audio send."
        )
        return HookResult(strategy=HookStrategy.CANCEL)

    def _is_voice_message(self, params: Any) -> bool:
        """
        Определяет, является ли отправляемое сообщение голосовым (ГС).
        Голосовые сообщения Telegram записываются в формате Ogg Opus (.ogg / .opus).
        """
        # Если флаг голосового явно выставлен
        if getattr(params, "isVoice", False) or getattr(params, "voice", False):
            return True

        path = getattr(params, "path", None)
        if not path or not isinstance(path, str):
            return False

        lower_path = path.lower()
        if not (lower_path.endswith(".ogg") or lower_path.endswith(".opus")):
            return False

        # Исключаем явную музыку или фото
        if getattr(params, "music", False) or getattr(params, "isMusic", False):
            return False

        if getattr(params, "photo", None) is not None:
            return False

        return True

    def _extract_reply_to_msg_id(self, params: Any) -> Optional[int]:
        """
        Извлекает ID сообщения, на которое был сделан ответ (reply),
        поддерживая различные версии моделей Telegram.
        """
        # 1. replyToMsg
        reply_obj = getattr(params, "replyToMsg", None)
        if reply_obj is not None:
            if isinstance(reply_obj, int):
                return reply_obj
            if hasattr(reply_obj, "getId"):
                try:
                    res = int(reply_obj.getId())
                    if res > 0:
                        return res
                except Exception:
                    pass
            if hasattr(reply_obj, "id"):
                try:
                    res = int(reply_obj.id)
                    if res > 0:
                        return res
                except Exception:
                    pass

        # 2. replyingMessageObject
        msg_obj = getattr(params, "replyingMessageObject", None)
        if msg_obj is not None:
            if hasattr(msg_obj, "getId"):
                try:
                    res = int(msg_obj.getId())
                    if res > 0:
                        return res
                except Exception:
                    pass
            if hasattr(msg_obj, "id"):
                try:
                    res = int(msg_obj.id)
                    if res > 0:
                        return res
                except Exception:
                    pass

        # 3. replyTo (TLRPC.InputReplyTo)
        input_reply = getattr(params, "replyTo", None)
        if input_reply is not None:
            if hasattr(input_reply, "reply_to_msg_id"):
                try:
                    res = int(input_reply.reply_to_msg_id)
                    if res > 0:
                        return res
                except Exception:
                    pass

        return None

    def _process_and_send(
        self,
        account: int,
        peer_id: int,
        reply_to_msg_id: Optional[int],
        audio_bytes: Optional[bytes],
        file_path: Optional[str],
        api_key: str,
        model: str,
    ):
        """
        Фоновая процедура:
        1. Читает аудио (если ещё не считано).
        2. Отправляет запрос в Groq Whisper API.
        3. Получает текст.
        4. Отправляет текстовое сообщение в чат.
        """
        try:
            if not audio_bytes and file_path:
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        audio_bytes = f.read()

            if not audio_bytes:
                self.logger.error("Audio data is empty, cannot transcribe")
                BulletinHelper.show_error("Groq Whisper: не удалось прочитать аудиофайл")
                return

            self.logger.info(f"Transcribing audio ({len(audio_bytes)} bytes) using model '{model}'...")

            text = transcribe_groq_whisper(
                audio_bytes=audio_bytes,
                api_key=api_key,
                model=model,
            )

            if not text:
                self.logger.info("Groq Whisper returned empty text (speech not detected)")
                BulletinHelper.show_info("Groq Whisper: речь не распознана")
                return

            self.logger.info(f"Transcription successful: {text[:50]}... Sending message...")

            self._send_text_result(account, peer_id, text, reply_to_msg_id)

        except Exception as e:
            self.logger.exception("Error in Groq Whisper processing")
            err_msg = str(e)
            if len(err_msg) > 90:
                err_msg = err_msg[:87] + "..."
            BulletinHelper.show_error(f"Groq Whisper: {err_msg}")

    def _send_text_result(
        self,
        account: int,
        peer_id: int,
        text: str,
        reply_to_msg_id: Optional[int],
    ):
        """
        Отправляет текстовое сообщение в чат с правильным аккаунтом и реплаем.
        """
        sent = False

        # Попытка 1: через account-bound client
        try:
            client = self.client(account)
            if hasattr(client, "send_text"):
                if reply_to_msg_id:
                    client.send_text(peer_id, text, replyToMsg=reply_to_msg_id)
                else:
                    client.send_text(peer_id, text)
                sent = True
        except Exception as e:
            self.logger.warning(f"self.client({account}).send_text failed: {e}")

        # Попытка 2: fallback через client_utils.send_text
        if not sent:
            try:
                if reply_to_msg_id:
                    send_text(peer_id, text, replyToMsg=reply_to_msg_id, account=account)
                else:
                    send_text(peer_id, text, account=account)
                sent = True
            except Exception as e:
                self.logger.error(f"client_utils.send_text fallback failed: {e}")
                raise
