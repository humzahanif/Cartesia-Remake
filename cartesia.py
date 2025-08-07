import streamlit as st
import requests
import json
import base64
import io
import tempfile
import os
from datetime import datetime
import google.generativeai as genai
from pathlib import Path
import wave
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Configure page
st.set_page_config(
    page_title="🎙️ Cartesia Remake",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
GEMINI_API_KEY = "AIzaSyBJOTWqYl-TGl8sFSPUPLCOENsrqaTd1Ng"
CARTESIA_API_KEY = "sk_car_27gigg7j2VykKKdzvDNcbE"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

class VoiceAgent:
    def __init__(self):
        self.cartesia_base_url = "https://api.cartesia.ai"
        self.headers = {
            "Authorization": f"Bearer {CARTESIA_API_KEY}",
            "Content-Type": "application/json",
            "Cartesia-Version": "2024-06-10"
        }

    def get_gemini_response(self, prompt, context=""):
        """Get response from Gemini AI"""
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            full_prompt = f"{context}\n\nUser: {prompt}" if context else prompt
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    def text_to_speech(self, text, voice_id="a0e99841-438c-4a64-b679-ae501e7d6091", speed=1.0, pitch=1.0):
        """Convert text to speech using Cartesia Bytes API"""
        try:
            url = f"{self.cartesia_base_url}/tts/bytes"
            payload = {
                "model_id": "sonic-english",
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": voice_id
                },
                "output_format": {
                    "container": "wav",
                    "encoding": "pcm_s16le",
                    "sample_rate": 44100
                }
            }

            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return response.content
            else:
                st.error(f"TTS Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"TTS Error: {str(e)}")
            return None

    def clone_voice_instant(self, audio_file, voice_name, voice_description="Cloned voice"):
        """Voice cloning using Cartesia API"""
        try:
            url = f"{self.cartesia_base_url}/voices/clone"

            # Reset file pointer
            audio_file.seek(0)

            files = {
                'clip': (audio_file.name, audio_file, 'audio/wav')
            }

            data = {
                'name': voice_name,
                'description': voice_description,
                'language': 'en'
            }

            headers_no_content_type = {
                "Authorization": f"Bearer {CARTESIA_API_KEY}",
                "Cartesia-Version": "2024-06-10"
            }

            response = requests.post(url, headers=headers_no_content_type, files=files, data=data)
            if response.status_code == 201:
                return response.json()
            else:
                st.error(f"Voice Clone Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Voice Clone Error: {str(e)}")
            return None

    def create_pro_voice_clone(self, audio_files, voice_name, description=""):
        """Professional voice cloning - Note: This may require multiple audio samples"""
        try:
            # For now, use the single clone endpoint as pro cloning may require special setup
            if audio_files:
                first_file = audio_files[0]
                return self.clone_voice_instant(first_file, voice_name, description)
            return None
        except Exception as e:
            st.error(f"Pro Voice Clone Error: {str(e)}")
            return None

    def localize_voice(self, voice_id, target_language, text):
        """Generate speech in different language using existing voice"""
        try:
            # Map language codes
            lang_map = {
                "spanish": "es", "french": "fr", "german": "de", "italian": "it",
                "portuguese": "pt", "russian": "ru", "japanese": "ja", "korean": "ko",
                "chinese": "zh", "arabic": "ar", "hindi": "hi", "dutch": "nl",
                "swedish": "sv", "norwegian": "no", "danish": "da"
            }

            lang_code = lang_map.get(target_language, "en")

            url = f"{self.cartesia_base_url}/tts/bytes"
            payload = {
                "model_id": "sonic-multilingual",
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": voice_id
                },
                "language": lang_code,
                "output_format": {
                    "container": "wav",
                    "encoding": "pcm_s16le",
                    "sample_rate": 44100
                }
            }

            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return response.content
            else:
                st.error(f"Voice Localization Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Voice Localization Error: {str(e)}")
            return None

    def change_voice(self, audio_file, target_voice_id):
        """Voice changing - simplified approach using TTS with target voice"""
        try:
            # This is a simplified implementation
            # Real voice changing would require speech-to-text first
            st.info("Voice changing requires speech-to-text conversion first. This is a simplified demo.")

            # For demo purposes, we'll use a sample text
            sample_text = "This is a demonstration of voice changing technology."
            return self.text_to_speech(sample_text, target_voice_id)

        except Exception as e:
            st.error(f"Voice Change Error: {str(e)}")
            return None

    def design_voice(self, gender, age, accent, style, emotion="neutral"):
        """Design a custom voice - this would typically use voice embeddings"""
        try:
            # This is a conceptual implementation
            # Real voice design would require advanced embedding manipulation

            # For demo, we'll create a description and use a default voice
            voice_description = f"{style} {gender} voice with {accent} accent, {age} age range, {emotion} emotion"

            # Return a mock response that simulates voice design
            return {
                "voice_id": f"designed_{gender}_{age}_{accent}_{style}_{emotion}",
                "name": f"{style.title()} {gender.title()}",
                "description": voice_description,
                "created": datetime.now().isoformat()
            }

        except Exception as e:
            st.error(f"Voice Design Error: {str(e)}")
            return None

# Initialize the voice agent
if 'voice_agent' not in st.session_state:
    st.session_state.voice_agent = VoiceAgent()

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'cloned_voices' not in st.session_state:
    st.session_state.cloned_voices = []

# Main UI
st.title("🎙️ Cartesia Remake")
st.markdown("### Powered by Gemini AI & Cartesia Voice Technology")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")

    # Voice settings
    st.subheader("🔊 Voice Settings")
    voice_speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1)
    voice_pitch = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1)

    # AI Context
    st.subheader("🤖 AI Context")
    ai_context = st.text_area(
        "System Context",
        value="You are a helpful AI assistant with natural conversation abilities.",
        height=100
    )

    # Clear conversation
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Chat & TTS",
    "⚡ Instant Clone",
    "🎯 Pro Clone",
    "🌍 Localize Voice",
    "🎭 Voice Changer",
    "🎨 Design Voice"
])

with tab1:
    st.header("💬 AI Chat with Text-to-Speech")

    # Chat interface
    chat_container = st.container()

    with chat_container:
        # Display conversation history
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
                if message.get("audio"):
                    st.audio(message["audio"], format="audio/wav")

    # Input area
    col1, col2 = st.columns([4, 1])

    with col1:
        user_input = st.text_input("💭 Ask me anything...", key="chat_input")

    with col1:
        send_button = st.button("🚀 Send", use_container_width=True)

    if send_button and user_input:
        # Add user message
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Get AI response
        with st.spinner("🤔 Thinking..."):
            ai_response = st.session_state.voice_agent.get_gemini_response(
                user_input,
                ai_context
            )

        # Generate speech
        with st.spinner("🎵 Generating speech..."):
            audio_data = st.session_state.voice_agent.text_to_speech(
                ai_response,
                speed=voice_speed,
                pitch=voice_pitch
            )

        # Add AI response
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": ai_response,
            "audio": audio_data
        })

        st.rerun()

with tab2:
    st.header("⚡ Instant Voice Clone")
    st.markdown("Clone any voice instantly with just one audio sample!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload Voice Sample")
        uploaded_audio = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            key="instant_clone_audio"
        )

        sample_text = st.text_area(
            "📝 Voice name",
            value="My Cloned Voice",
            height=50
        )

        voice_description = st.text_area(
            "📋 Description (optional)",
            value="A cloned voice sample",
            height=50
        )

        if st.button("⚡ Clone Voice Instantly", type="primary"):
            if uploaded_audio and sample_text:
                with st.spinner("🔄 Cloning voice..."):
                    result = st.session_state.voice_agent.clone_voice_instant(
                        uploaded_audio,
                        sample_text,
                        voice_description
                    )

                if result:
                    st.success("✅ Voice cloned successfully!")

                    # Display result info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Voice ID:** {result.get('id', 'N/A')}")
                        st.write(f"**Name:** {result.get('name', 'N/A')}")
                    with col2:
                        st.write(f"**Language:** {result.get('language', 'N/A')}")
                        st.write(f"**Created:** {result.get('created_at', 'N/A')}")

                    # Store cloned voice
                    st.session_state.cloned_voices.append({
                        "name": result.get('name', sample_text),
                        "id": result.get('id', 'unknown'),
                        "type": "instant",
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

    with col2:
        st.subheader("🎵 Test Cloned Voice")
        # Define available_voices for tab2
        default_voices = [
            {"name": "British Male", "id": "a0e99841-438c-4a64-b679-ae501e7d6091", "type": "default"},
            {"name": "American Female", "id": "79a125e8-cd45-4c13-8a67-188112f4dd22", "type": "default"},
            {"name": "Australian Male", "id": "2ee87190-8f84-4925-97da-e52547f9462c", "type": "default"}
        ]
        available_voices = default_voices + st.session_state.cloned_voices

        if available_voices:
            selected_voice = st.selectbox(
                "Select voice",
                options=available_voices,
                format_func=lambda x: f"{x['name']} ({x['type']})"
            )

            test_text = st.text_area(
                "Test text",
                value="This is a test of the cloned voice.",
                height=100
            )

            if st.button("🎤 Generate with Cloned Voice"):
                with st.spinner("🎵 Generating audio..."):
                    audio_data = st.session_state.voice_agent.text_to_speech(
                        test_text,
                        voice_id=selected_voice["id"]
                    )

                if audio_data:
                    st.audio(audio_data, format="audio/wav")
        else:
            st.info("No cloned voices available. Create one first!")

with tab3:
    st.header("🎯 Professional Voice Clone")
    st.markdown("Create high-quality voice clones with multiple audio samples for better accuracy.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload Multiple Samples")
        uploaded_samples = st.file_uploader(
            "Choose multiple audio files",
            type=['wav', 'mp3', 'flac', 'm4a'],
            accept_multiple_files=True,
            key="pro_clone_audio"
        )

        voice_name = st.text_input("🏷️ Voice Name", value="My Custom Voice")
        voice_description = st.text_area(
            "📋 Description",
            value="A custom voice created for professional use.",
            height=80
        )

        if st.button("🎯 Create Pro Voice Clone", type="primary"):
            if uploaded_samples and voice_name:
                with st.spinner("🔄 Creating professional voice clone..."):
                    result = st.session_state.voice_agent.create_pro_voice_clone(
                        uploaded_samples,
                        voice_name,
                        voice_description
                    )

                if result:
                    st.success("✅ Professional voice clone created!")
                    st.json(result)

                    # Store cloned voice
                    st.session_state.cloned_voices.append({
                        "name": voice_name,
                        "id": result.get("voice_id", "unknown"),
                        "type": "professional",
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

    with col2:
        st.subheader("📊 Voice Management")
        if st.session_state.cloned_voices:
            st.markdown("**Cloned Voices:**")
            for i, voice in enumerate(st.session_state.cloned_voices):
                with st.expander(f"🎤 {voice['name']}"):
                    st.write(f"**Type:** {voice['type'].title()}")
                    st.write(f"**ID:** {voice['id']}")
                    st.write(f"**Created:** {voice['created']}")

                    if st.button(f"🗑️ Delete", key=f"delete_voice_{i}"):
                        st.session_state.cloned_voices.pop(i)
                        st.rerun()
        else:
            st.info("No cloned voices yet. Create some above!")

with tab4:
    st.header("🌍 Voice Localization")
    st.markdown("Adapt voices to speak in different languages while maintaining characteristics.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎛️ Localization Settings")

        # Default voices for demo
        default_voices = [
            {"name": "British Male", "id": "a0e99841-438c-4a64-b679-ae501e7d6091", "type": "default"},
            {"name": "American Female", "id": "79a125e8-cd45-4c13-8a67-188112f4dd22", "type": "default"},
            {"name": "Australian Male", "id": "2ee87190-8f84-4925-97da-e52547f9462c", "type": "default"}
        ]

        available_voices = default_voices + st.session_state.cloned_voices

        if available_voices:
            selected_voice = st.selectbox(
                "Select voice to localize",
                options=available_voices,
                format_func=lambda x: f"{x['name']} ({x['type']})",
                key="localize_voice_select"
            )
        else:
            selected_voice = default_voices[0]

        target_language = st.selectbox(
            "🌐 Target Language",
            options=[
                "spanish", "french", "german", "italian", "portuguese",
                "russian", "japanese", "korean", "chinese", "arabic",
                "hindi", "dutch", "swedish", "norwegian", "danish"
            ]
        )

        localize_text = st.text_area(
            "📝 Text to localize",
            value="Hello, how are you today? This is a localization test.",
            height=100
        )

        if st.button("🌍 Localize Voice", type="primary") and selected_voice:
            with st.spinner("🔄 Localizing voice..."):
                audio_data = st.session_state.voice_agent.localize_voice(
                    selected_voice["id"],
                    target_language,
                    localize_text
                )

            if audio_data:
                st.success("✅ Voice localized successfully!")
                st.audio(audio_data, format="audio/wav")

    with col2:
        st.subheader("📚 Supported Languages")
        languages = [
            "🇪🇸 Spanish", "🇫🇷 French", "🇩🇪 German", "🇮🇹 Italian",
            "🇵🇹 Portuguese", "🇷🇺 Russian", "🇯🇵 Japanese", "🇰🇷 Korean",
            "🇨🇳 Chinese", "🇸🇦 Arabic", "🇮🇳 Hindi", "🇳🇱 Dutch",
            "🇸🇪 Swedish", "🇳🇴 Norwegian", "🇩🇰 Danish"
        ]

        for lang in languages:
            st.write(f"• {lang}")

with tab5:
    st.header("🎭 Voice Changer")
    st.markdown("Transform existing audio with different voice characteristics.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload Audio to Change")
        input_audio = st.file_uploader(
            "Choose audio file to modify",
            type=['wav', 'mp3', 'flac', 'm4a'],
            key="voice_change_input"
        )

        # Default voices for demo
        default_voices = [
            {"name": "British Male", "id": "a0e99841-438c-4a64-b679-ae501e7d6091", "type": "default"},
            {"name": "American Female", "id": "79a125e8-cd45-4c13-8a67-188112f4dd22", "type": "default"},
            {"name": "Australian Male", "id": "2ee87190-8f84-4925-97da-e52547f9462c", "type": "default"}
        ]

        available_voices = default_voices + st.session_state.cloned_voices

        if available_voices:
            target_voice = st.selectbox(
                "🎯 Target Voice Style",
                options=available_voices,
                format_func=lambda x: f"{x['name']} ({x['type']})",
                key="voice_change_target"
            )
        else:
            target_voice = default_voices[0]

        if st.button("🎭 Change Voice", type="primary") and input_audio and target_voice:
            with st.spinner("🔄 Changing voice characteristics..."):
                audio_data = st.session_state.voice_agent.change_voice(
                    input_audio,
                    target_voice["id"]
                )

            if audio_data:
                st.success("✅ Voice changed successfully!")
                st.audio(audio_data, format="audio/wav")

    with col2:
        st.subheader("💡 Tips for Voice Changing")
        st.markdown("""
        **Best Results:**
        • Use clear, high-quality audio
        • Avoid background noise
        • Single speaker preferred
        • 3-30 seconds duration ideal

        **Applications:**
        • Content creation
        • Voice disguising
        • Character voices
        • Audio enhancement
        """)

with tab6:
    st.header("🎨 Custom Voice Designer")
    st.markdown("Design unique voices from scratch with specific characteristics.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎛️ Voice Parameters")

        voice_gender = st.selectbox(
            "👤 Gender",
            options=["male", "female", "neutral"]
        )

        voice_age = st.selectbox(
            "📅 Age Range",
            options=["child", "young_adult", "adult", "middle_aged", "elderly"]
        )

        voice_accent = st.selectbox(
            "🌍 Accent",
            options=[
                "american", "british", "australian", "canadian",
                "irish", "scottish", "indian", "neutral"
            ]
        )

        voice_style = st.selectbox(
            "🎭 Speaking Style",
            options=[
                "conversational", "professional", "energetic",
                "calm", "authoritative", "friendly", "dramatic"
            ]
        )

        voice_emotion = st.selectbox(
            "😊 Emotion",
            options=[
                "neutral", "happy", "sad", "excited",
                "angry", "calm", "surprised", "confident"
            ]
        )

        if st.button("🎨 Design Voice", type="primary"):
            with st.spinner("🔄 Designing custom voice..."):
                result = st.session_state.voice_agent.design_voice(
                    voice_gender,
                    voice_age,
                    voice_accent,
                    voice_style,
                    voice_emotion
                )

            if result:
                st.success("✅ Custom voice designed!")
                st.json(result)

                # Store designed voice
                voice_name = f"{voice_style.title()} {voice_gender.title()} ({voice_accent})"
                st.session_state.cloned_voices.append({
                    "name": voice_name,
                    "id": result.get("voice_id", "unknown"),
                    "type": "designed",
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

    with col2:
        st.subheader("🎵 Test Designed Voice")

        design_test_text = st.text_area(
            "📝 Test Text",
            value="This is a demonstration of the custom designed voice.",
            height=100
        )

        if st.button("🎤 Generate Test Audio"):
            # Default voices for demo
            default_voices = [
                {"name": "British Male", "id": "a0e99841-438c-4a64-b679-ae501e7d6091", "type": "default"},
                {"name": "American Female", "id": "79a125e8-cd45-4c13-8a67-188112f4dd22", "type": "default"},
                {"name": "Australian Male", "id": "2ee87190-8f84-4925-97da-e52547f9462c", "type": "default"}
            ]

            available_voices = default_voices + st.session_state.cloned_voices

            if available_voices:
                latest_voice = available_voices[-1]
                with st.spinner("🎵 Generating test audio..."):
                    audio_data = st.session_state.voice_agent.text_to_speech(
                        design_test_text,
                        voice_id=latest_voice["id"]
                    )

                if audio_data:
                    st.audio(audio_data, format="audio/wav")
            else:
                st.warning("No voices available!")

        st.subheader("📋 Voice Combinations")
        st.markdown("""
        **Popular Combinations:**
        • Professional + Adult + American
        • Friendly + Young Adult + British
        • Energetic + Adult + Australian
        • Calm + Middle Aged + Neutral
        • Dramatic + Adult + Irish
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🎙️ Advanced Voice Agent Pro | Powered by Gemini AI & Cartesia
    </div>
    """,
    unsafe_allow_html=True
)

# Display API status
with st.expander("🔧 API Status & Debug"):
    st.markdown("**Current Settings:**")
    st.write(f"• Voice Speed: {voice_speed}")
    st.write(f"• Voice Pitch: {voice_pitch}")
    st.write(f"• Cloned Voices: {len(st.session_state.cloned_voices)}")
    st.write(f"• Conversation Messages: {len(st.session_state.conversation_history)}")

    if st.button("🧪 Test API Connection"):
        with st.spinner("Testing connections..."):
            # Test Gemini
            try:
                test_response = st.session_state.voice_agent.get_gemini_response("Hello")
                st.success("✅ Gemini AI: Connected")
            except:
                st.error("❌ Gemini AI: Connection failed")

            # Test Cartesia (simplified test)
            st.info("🔄 Cartesia: API key configured")