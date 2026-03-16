from gtts import gTTS
import os
import tempfile
import uuid

def text_to_speech(text, language='en'):
    """
    Converts text to speech and returns the path to the audio file.
    Uses uuid to generate unique filenames to avoid locking issues.
    """
    try:
        # Map common language names to gTTS language codes
        lang_map = {
            'English': 'en',
            'Telugu': 'te',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Hindi': 'hi',
            'Italian': 'it',
            'Portuguese': 'pt'
        }
        
        lang_code = lang_map.get(language, 'en')
        
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        # Use a standard temp directory but with unique filenames
        temp_dir = tempfile.gettempdir()
        filename = f"tts_{uuid.uuid4()}.mp3"
        file_path = os.path.join(temp_dir, filename)
        
        tts.save(file_path)
        return file_path
        
    except Exception as e:
        print(f"Error in TTS: {e}")
        return None
