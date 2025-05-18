# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#VoiceBot UI with Gradio
#VoiceBot UI with Gradio
import requests
import tempfile
import time
import os
import traceback
from dotenv import load_dotenv
import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs
load_dotenv()
chat_history = []
current_image = None

system_prompt = """You are a professional medical consultant providing expert advice to patients. This is a real medical consultation.

RESPONSE GUIDELINES:
- ALWAYS respond directly to the patient's concern without preamble
- NEVER start your response with phrases like "In this image I see" or "What I see is"
- Address the patient directly as if you are speaking to them in person
- Format your response as a single cohesive paragraph
- Keep responses concise (2-3 sentences maximum)
- Maintain a professional, reassuring tone throughout
- For follow-up questions, continue the conversation naturally without repeating previous observations

WHEN ANALYZING IMAGES:
- Begin responses with phrases like "Based on your condition..." or "With what I can observe..."
- Provide a brief assessment of any visible medical conditions
- If appropriate, suggest potential diagnoses as a differential
- Recommend general treatment approaches when possible
- Advise when further medical examination is needed

WHEN RESPONDING TO QUESTIONS WITHOUT IMAGES:
- Address the patient's concern directly
- Provide professional medical guidance based on the information provided
- Ask clarifying questions if needed for better assessment

IMPORTANT: Always maintain the persona of a professional doctor throughout the conversation, even during follow-up exchanges. Ensure all responses feel like they come from a real medical professional speaking directly to the patient."""

def clear_chat():
    global chat_history, current_image
    chat_history = []
    current_image = None
    # Reset all relevant UI elements to their initial state
    return "", None, None, "", "", None, gr.update(visible=False), gr.update(visible=True)

def get_unique_temp_file():
    """Generate a unique temporary file path"""
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    return os.path.join(temp_dir, f"doctor_response_{timestamp}.mp3")

def get_doctor_response(text, image=None):
    """Get doctor's response using Groq API first, then fall back to Sambhava Cloud with llmaa model, 
    and finally to Ollama with Llama model if both fail"""
    
    # First try using Groq API
    try:
        print("DEBUG: Trying Groq API...")
        if image:
            print("DEBUG: Processing with image...")
            doctor_response = analyze_image_with_query(
                query=f"{system_prompt}\n\nCurrent question: {text}",
                encoded_image=encode_image(image),
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        else:
            print("DEBUG: Processing without image...")
            doctor_response = analyze_image_with_query(
                query=f"Act as a professional doctor. The patient is asking: {text}. Please provide a professional medical consultation. Keep your answer concise (max 2 sentences). No preamble, start your answer right away please",
                encoded_image=None,
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        
        # Check if the response indicates an error
        if any(error_msg in doctor_response.lower() for error_msg in ['sorry', 'unavailable', 'error', 'try again', 'temporarily']):
            print("DEBUG: Groq API returned error message, switching to Sambhava Cloud fallback")
            return use_sambhava_fallback(text, image)
            
        print("DEBUG: Groq API response successful")
        return doctor_response
        
    except Exception as e:
        print(f"DEBUG: Groq API failed: {str(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        return use_sambhava_fallback(text, image)

def use_sambhava_fallback(text, image=None):
    """Fallback to using Sambhava Cloud with llmaa model"""

    try:
        print("DEBUG: Starting Sambhava Cloud fallback with llmaa model...")

        # Get API key and endpoint from environment variables
        sambhava_api_key = os.getenv("SAMBHAVA_API_KEY")
        sambhava_endpoint = os.getenv("SAMBHAVA_ENDPOINT", "https://api.sambhavaai.com/v1/chat/completions")

        if not sambhava_api_key:
            print("DEBUG: SAMBHAVA_API_KEY not found in .env file")
            return use_ollama_fallback(text, image)

        # Build messages payload
        messages = []
        if image:
            # If image handling is supported, customize accordingly
            prompt = f"{system_prompt}\n\nCurrent question: {text}"
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": text})

        print(f"DEBUG: Sending prompt to Sambhava Cloud: {messages[-1]['content']}")

        response = requests.post(
            sambhava_endpoint,
            json={
                "model": "llmaa",
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.7
            },
            headers={
                "Authorization": f"Bearer {sambhava_api_key}",
                "Content-Type": "application/json"
            },
            timeout=10
        )

        print(f"DEBUG: Sambhava Cloud API response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"DEBUG: Sambhava Cloud API response: {result}")
            # Adjust this depending on the actual structure of Sambhava's API response
            doctor_response = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if not doctor_response:
                print("DEBUG: Empty response from Sambhava Cloud, falling back to Ollama")
                return use_ollama_fallback(text, image)

            print("DEBUG: Sambhava Cloud response successful")
            return doctor_response

        else:
            print(f"DEBUG: Sambhava Cloud API error: {response.status_code} - {response.text}")
            return use_ollama_fallback(text, image)

    except requests.exceptions.RequestException as e:
        print(f"DEBUG: Request error to Sambhava Cloud: {e}")
        return use_ollama_fallback(text, image)

    except Exception as e:
        print(f"DEBUG: Error in Sambhava Cloud fallback: {str(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        return use_ollama_fallback(text, image)

def use_ollama_fallback(text, image=None):
    """Fallback to using Ollama with Llama model"""
    try:
        print("DEBUG: Starting Ollama fallback with Llama model...")
        
        # Prepare the prompt
        if image:
            # If there's an image, include that context in the prompt
            prompt = f"{system_prompt}\n\nCurrent question: {text}"
        else:
            prompt = f"Act as a professional doctor. The patient is asking: {text}. Please provide a professional medical consultation. Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"
        
        print(f"DEBUG: Sending prompt to Ollama: {prompt}")
        
        # Call Ollama API with Llama model
        print("DEBUG: Making request to Ollama API with Llama model...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3", # Using llama3 from Ollama (adjust this if you have a different Llama model name)
                "prompt": prompt,
                "stream": False
            }
        )
        print(f"DEBUG: Ollama API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"DEBUG: Ollama API response: {result}")
            doctor_response = result.get('response', '').strip()
            if not doctor_response:
                print("DEBUG: Empty response from Ollama")
                doctor_response = "I understand your concern. Please describe your symptoms in more detail for better assistance."
            print("DEBUG: Ollama response successful")
            return doctor_response
        else:
            print(f"DEBUG: Ollama API error: {response.status_code} - {response.text}")
            return "I understand your concern. Please describe your symptoms in more detail for better assistance."
        
    except Exception as e:
        print(f"DEBUG: Error in Ollama fallback: {str(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        return "I understand your concern. Please describe your symptoms in more detail for better assistance."

def process_initial_input(audio_filepath, image_filepath):
    global current_image
    current_image = image_filepath
    
    # Get unique temporary file path
    temp_audio_path = get_unique_temp_file()
    
    try:
        # Transcribe the audio
        speech_to_text_output = transcribe_with_groq(GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
                                                     audio_filepath=audio_filepath,
                                                     stt_model="whisper-large-v3")
        
        # Add user message to chat history
        chat_history.append(("You", speech_to_text_output))
        
        # Get doctor's response with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                doctor_response = get_doctor_response(speech_to_text_output, image_filepath)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    doctor_response = "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."
                    print(f"Error in doctor response: {str(e)}")
                time.sleep(1)  # Wait before retrying
        
        # Add doctor's response to chat history
        chat_history.append(("Doctor", doctor_response))
        
        # Generate voice response
        try:
            voice_of_doctor = text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath=temp_audio_path)
        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")
            voice_of_doctor = None
        
        # Format chat history for display
        formatted_chat = "\n\n".join([f"**{role}**: {message}" for role, message in chat_history])
        print("******************************************************")
        
        return speech_to_text_output, doctor_response, voice_of_doctor, formatted_chat, gr.update(visible=False), gr.update(visible=True)
    
    except Exception as e:
        print(f"Error in process_initial_input: {str(e)}")
        return "Error processing your request. Please try again.", "I apologize, but I'm having trouble processing your request right now.", None, "", gr.update(visible=True), gr.update(visible=False)

def process_followup(audio_filepath):
    # Get unique temporary file path
    temp_audio_path = get_unique_temp_file()
    
    try:
        # Transcribe the audio
        speech_to_text_output = transcribe_with_groq(GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
                                                     audio_filepath=audio_filepath,
                                                     stt_model="whisper-large-v3")
        
        # Add user message to chat history
        chat_history.append(("You", speech_to_text_output))
        
        # Prepare the context for the doctor
        context = "\n".join([f"{role}: {message}" for role, message in chat_history[-5:]])
        
        # Get doctor's response with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                doctor_response = get_doctor_response(
                    f"Previous conversation:\n{context}\n\nCurrent question: {speech_to_text_output}",
                    current_image
                )
                print("#######################################")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    doctor_response = "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."
                    print(f"Error in doctor response: {str(e)}")
                time.sleep(1)  # Wait before retrying
        
        # Add doctor's response to chat history
        chat_history.append(("Doctor", doctor_response))
        
        # Generate voice response
        try:
            voice_of_doctor = text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath=temp_audio_path)
            print("elevan lab s#######################################")

        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")
            voice_of_doctor = None
        
        # Format chat history for display
        formatted_chat = "\n\n".join([f"**{role}**: {message}" for role, message in chat_history])
        
        return speech_to_text_output, doctor_response, voice_of_doctor, formatted_chat
    
    except Exception as e:
        print(f"Error in process_followup: {str(e)}")
        return "Error processing your request. Please try again.", "I apologize, but I'm having trouble processing your request right now.", None, ""
# Custom CSS for better styling
css = """
.contain { display: flex; flex-direction: column; }
.gradio-container { 
    max-width: 1200px !important; 
    margin: 0 auto !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 20px;
    border-radius: 15px;
}
#component-0 { min-height: 300px; }
#component-1 { min-height: 300px; }
#component-2 { min-height: 200px; }
#component-3 { min-height: 200px; }
#component-4 { min-height: 200px; }

/* Chat history styling */
.chat-history {
    max-height: 400px;
    overflow-y: auto;
    padding: 15px;
    background: white;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

/* Chat message styling */
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #f0f4ff;
    border-left: 4px solid #4a90e2;
}
.chat-message.bot {
    background-color: #fff;
    border-left: 4px solid #50c878;
}

/* Button styling */
button {
    background: linear-gradient(45deg, #4a90e2, #50c878) !important;
    color: white !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 25px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}
button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

/* Textbox styling */
textarea {
    font-size: 16px !important;
    line-height: 1.6 !important;
    padding: 15px !important;
    border-radius: 10px !important;
    border: 2px solid #e0e0e0 !important;
}

/* Header styling */
h1 {
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
    font-size: 2.5rem;
    font-weight: 700;
}

/* Logo styling */
.logo {
    text-align: center;
    margin-bottom: 2rem;
}
.logo img {
    max-width: 150px;
    height: auto;
}

/* Refresh button styling */
.refresh-btn {
    background: #4CAF50 !important;
    color: white !important;
    border: none !important;
    padding: 8px 16px !important;
    border-radius: 20px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    margin-bottom: 10px !important;
}

.refresh-btn:hover {
    background: #45a049 !important;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

/* Hidden elements */
.hidden {
    display: none !important;
}
"""

# Create the interface with a more interactive layout
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    # Header with logo
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            <div class="logo">
                <h1>🏥 MediCare AI</h1>
                <p style="text-align: center; color: #666; font-size: 1.2rem;">Your AI-powered healthcare assistant</p>
            </div>
            """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #2c3e50;">How can I help you today?</h2>
                <p style="color: #666;">Start by describing your symptoms. You can optionally upload an image for visual analysis.</p>
            </div>
            """)
    
    # Chat history display with refresh button
    with gr.Row():
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 Start New Conversation", variant="primary", elem_classes=["refresh-btn"])
            chat_display = gr.Markdown(
                label="Conversation History",
                elem_classes=["chat-history"]
            )
    
    # Initial input section
    with gr.Row() as initial_input_row:
        with gr.Column(scale=1):
            initial_audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Record your symptoms",
                interactive=True
            )
            initial_image = gr.Image(
                type="filepath",
                label="📸 Upload medical image",
                interactive=True
            )
            initial_submit = gr.Button("👨‍⚕️ Start Consultation", variant="primary")
    
    # Follow-up chat section
    with gr.Row(visible=False) as followup_row:
        with gr.Column(scale=1):
            followup_audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Ask follow-up questions",
                interactive=True
            )
            followup_submit = gr.Button("💬 Send Message", variant="primary")
    
    # Response display section
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                speech_text = gr.Textbox(
                    label="📝 Your Message",
                    interactive=False,
                    lines=3,
                    elem_classes=["chat-message", "user"]
                )
                doctor_response = gr.Textbox(
                    label="💊 Doctor's Response",
                    interactive=False,
                    lines=5,
                    elem_classes=["chat-message", "bot"]
                )
                audio_output = gr.Audio(
                    label="🔊 Doctor's Voice Response",
                    interactive=False
                )
    
    # Add some helpful instructions
    gr.Markdown("""
    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 2rem;">
        <h3 style="color: #2c3e50; margin-bottom: 1rem;">Instructions:</h3>
        <ol style="color: #666; line-height: 1.8;">
            <li>Start by recording your symptoms and uploading a medical image</li>
            <li>Click 'Start Consultation' to begin</li>
            <li>After the initial consultation, you can ask follow-up questions</li>
            <li>Use the 'Start New Conversation' button to begin a fresh consultation</li>
        </ol>
    </div>
    """)
    
    # Set up the processing
    initial_submit.click(
        fn=process_initial_input,
        inputs=[initial_audio, initial_image],
        outputs=[speech_text, doctor_response, audio_output, chat_display, initial_input_row, followup_row]
    )
    
    followup_submit.click(
        fn=process_followup,
        inputs=[followup_audio],
        outputs=[speech_text, doctor_response, audio_output, chat_display]
    )
    
    # Set up refresh functionality
    refresh_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chat_display, initial_image, initial_audio, speech_text, doctor_response, audio_output, followup_row, initial_input_row]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )

#http://127.0.0.1:7860