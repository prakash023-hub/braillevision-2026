"""
BrailleVision — Text-to-Speech output
Tries gTTS (online) first, falls back to pyttsx3 (offline).
"""

import os
import tempfile
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def text_to_speech(text: str) -> Optional[str]:
    """
    Convert text to speech audio file.
    Returns path to .mp3/.wav file, or None on failure.
    """
    if not text or not text.strip():
        return None

    # Try gTTS (online, better quality)
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang='en', slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        logger.warning(f"gTTS failed: {e}, trying pyttsx3")

    # Fallback: pyttsx3 (offline)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        engine.save_to_file(text, tmp.name)
        engine.runAndWait()
        return tmp.name
    except Exception as e:
        logger.error(f"pyttsx3 also failed: {e}")
        return None
