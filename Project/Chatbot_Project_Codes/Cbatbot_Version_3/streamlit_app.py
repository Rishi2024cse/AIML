import streamlit as st
import json
import random
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# ===============================================
# 1. INITIAL SETUP & STATE MANAGEMENT
# ===============================================

def load_phq9_data():
    """Loads PHQ-9 data from the JSON file with fallback."""
    try:
        with open('phq9_questions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Error: phq9_questions.json not found. Please ensure the file is in the project directory.")
        # Provide fallback data
        return {
            "questions": [
                "Little interest or pleasure in doing things",
                "Feeling down, depressed, or hopeless",
                "Trouble falling or staying asleep, or sleeping too much",
                "Feeling tired or having little energy",
                "Poor appetite or overeating",
                "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
                "Trouble concentrating on things, such as reading the newspaper or watching television",
                "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual",
                "Thoughts that you would be better off dead or of hurting yourself in some way"
            ],
            "scale": {"not at all": 0, "several days": 1, "more than half the days": 2, "nearly every day": 3},
            "interpretation": {
                "0-4": "Minimal depression",
                "5-9": "Mild depression",
                "10-14": "Moderate depression",
                "15-19": "Moderately severe depression",
                "20-27": "Severe depression"
            }
        }

# --- Session State Initialization (CRITICAL for Streamlit persistence) ---
if 'phq9_data' not in st.session_state:
    st.session_state.phq9_data = load_phq9_data()
    st.session_state.test_complete = False
    st.session_state.current_q = 0
    st.session_state.score = 0
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! I'm an Adaptive Mental Fitness Chatbot. To get started, I'll ask you a few questions to understand how you've been feeling."}]
    st.session_state.last_intent = None
    st.session_state.chat_active = True

# Check for data load
if st.session_state.phq9_data:
    PHQ9_QUESTIONS = st.session_state.phq9_data['questions']
    PHQ9_SCALE = st.session_state.phq9_data['scale']
    PHQ9_INTERPRETATION = st.session_state.phq9_data['interpretation']
else:
    st.stop() # Stop execution if data is missing

# ===============================================
# 2. KNOWLEDGE BASE & AI LOGIC
# ===============================================

# --- CRISIS PRIORITY 0 ---
CRISIS_KEYWORDS = ["kill myself", "end my life", "suicide", "hurt myself", "i want to die", "take my life", "want to die", "ending it all"]
CRISIS_RESPONSE = (
    "⚠️ **EMERGENCY WARNING** ⚠️\n\n"
    "I am an AI and cannot provide emergency help. Your safety is paramount. "
    "Please contact a professional immediately.\n\n"
    "**USA/CANADA:** Call or text **988** (Suicide & Crisis Lifeline)\n"
    "**UK:** Call **111** or text \"SHOUT\" to **85258**\n"
    "**International:** Find your local crisis line at [findahelpline.com](https://findahelpline.com)\n\n"
    "You are not alone, and there are people who want to help you right now."
)

# --- Intent-Specific Responses (Priority 2) ---
INTENT_RESPONSES = {
    "sleep": {
        "keywords": ["sleep", "insomnia", "waking up", "bedtime", "tired", "can't sleep", "awake", "exhausted"],
        "advice": [
            "Trouble sleeping is draining. Let's talk about **sleep hygiene**. Can you try turning off all screens an hour before bed tonight?",
            "A racing mind keeps us awake. Would you be willing to try a short **guided meditation for sleep** tonight?",
            "Sleep issues are common with stress. Have you tried establishing a consistent bedtime routine?",
            "When you can't sleep, try getting up and doing something calming (like reading a book) for 15 minutes, then returning to bed."
        ]
    },
    "anxiety/stress": {
        "keywords": ["anxious", "stress", "panic", "worry", "overwhelmed", "deadline", "nervous", "anxiety"],
        "advice": [
            "When anxiety hits, try the **5-4-3-2-1 Grounding Technique**. Can you name 5 things you can see right now?",
            "Let's try **Box Breathing**. Inhale for 4, hold for 4, exhale for 4, hold for 4. Do this four times.",
            "Stress can feel overwhelming. Would breaking this down into smaller steps help?",
            "Remember that this feeling is temporary. What's one small thing you can control right now?"
        ]
    },
    "self_esteem": {
        "keywords": ["worthless", "failure", "stupid", "bad about myself", "can't do anything right", "not good enough", "useless"],
        "advice": [
            "It sounds like your internal critic is very loud right now. Let's try **Cognitive Restructuring**. Can you list one piece of evidence that disproves your thought?",
            "You are not a failure. Let's list **three things you are genuinely good at** or proud of, no matter how small.",
            "Everyone has difficult moments. What would you say to a friend who felt this way?",
            "Our thoughts aren't always facts. Can you identify one positive quality you have?"
        ]
    },
    "depression": {
        "keywords": ["depressed", "hopeless", "empty", "nothing matters", "no point", "sad all the time"],
        "advice": [
            "Depression can make everything feel heavy. Have you been able to do one small thing for yourself today?",
            "When depression speaks, it lies. What's one tiny thing that usually brings you even a moment of peace?",
            "This sounds really difficult. Would going for a short walk or changing your environment help right now?",
            "Depression often isolates us. Is there someone you feel comfortable reaching out to today?"
        ]
    },
    "general": {
        "keywords": ["hello", "hi", "hey", "how are you", "what can you do", "help"],
        "advice": [
            "Hello! I'm here to listen and offer support. How are you feeling today?",
            "Hi there! I'm a mental fitness chatbot. You can share what's on your mind, and I'll do my best to help.",
            "Welcome! I'm here to provide mental health support. What would you like to talk about?"
        ]
    }
}

# --- General Sentiment Responses (Priority 3 - Fallback) ---
PSYCH_RESPONSES = {
    "negative": [
        "That sounds very heavy. It takes courage to share that. Can you tell me more about what's causing this intense feeling?",
        "I hear the pain in your words. Would you like to explore this feeling further?",
        "Thank you for trusting me with this. What's been the most challenging part for you?",
        "I can sense this is really difficult for you. How long have you been feeling this way?"
    ],
    "positive": [
        "That's wonderful news! It sounds like you've made some positive steps. What contributed most to that feeling?",
        "I'm glad to hear that! What helped you reach this positive place?",
        "That's great! How can you build on this positive momentum?",
        "Wonderful! It's important to celebrate these moments. What made this possible?"
    ],
    "neutral": [
        "Thank you for sharing that with me. Please continue, I am here to listen without judgment.",
        "I appreciate you telling me this. What else is on your mind?",
        "Thank you for opening up. How has this been affecting your daily life?",
        "I'm listening. Could you tell me more about that?"
    ]
}

def analyze_sentiment(text):
    """Uses VADER to categorize text sentiment."""
    vs = sia.polarity_scores(text)
    compound_score = vs['compound']
    if compound_score >= 0.05: return "positive"
    if compound_score <= -0.05: return "negative"
    return "neutral"

def get_psych_response(user_input):
    """
    Determines the chatbot's response based on the priority structure.
    """
    normalized_input = user_input.lower().strip()
    
    # PRIORITY 1: CRISIS CHECK
    if any(keyword in normalized_input for keyword in CRISIS_KEYWORDS):
        st.session_state.last_intent = 'crisis'
        return CRISIS_RESPONSE

    # PRIORITY 2: INTENT CHECK
    for intent, data in INTENT_RESPONSES.items():
        if any(keyword in normalized_input for keyword in data["keywords"]):
            st.session_state.last_intent = intent
            return random.choice(data["advice"])

    # PRIORITY 3: SENTIMENT CHECK
    sentiment = analyze_sentiment(user_input)
    st.session_state.last_intent = sentiment 
    return random.choice(PSYCH_RESPONSES[sentiment])

# ===============================================
# 3. UI FUNCTIONS (PHQ-9)
# ===============================================

def calculate_phq9_result():
    """Displays the final PHQ-9 score and interpretation."""
    score = st.session_state.score
    interpretation = "Unknown"
    
    for score_range, meaning in PHQ9_INTERPRETATION.items():
        range_parts = score_range.split('-')
        if len(range_parts) == 2:
            if int(range_parts[0]) <= score <= int(range_parts[1]):
                interpretation = meaning
                break
    
    st.subheader("📝 PHQ-9 Test Results")
    st.info(f"Your total score is: **{score}/27**\n\nInterpretation: **{interpretation}**")
    st.session_state.test_complete = True
    
    # Add a final message to the chat history to initiate conversation
    follow_up = f"Thank you for completing the assessment. Your score is {score}/27, which suggests **{interpretation.lower()}**. Remember, this is just a screening tool, not a diagnosis. I'm here to listen and support you. What would you like to talk about?"
    st.session_state.messages.append({"role": "assistant", "content": follow_up})
    
    st.rerun() # Rerun to show the chat interface

def handle_answer(score_value):
    """Updates score, moves to next question, or ends test."""
    st.session_state.score += score_value
    st.session_state.current_q += 1
    
    if st.session_state.current_q >= len(PHQ9_QUESTIONS):
        calculate_phq9_result()
    else:
        st.rerun()

def display_phq9_ui():
    """Renders the current PHQ-9 question and buttons."""
    
    q_index = st.session_state.current_q
    question_text = PHQ9_QUESTIONS[q_index]
    
    st.subheader(f"Question {q_index + 1} of {len(PHQ9_QUESTIONS)}")
    st.write(f"**{question_text}**")
    st.write("How often in the last two weeks have you been bothered by this?")

    # Create buttons for each option
    for label, score in PHQ9_SCALE.items():
        if st.button(f"{label.capitalize()} ({score} point{'s' if score != 1 else ''})", 
                    key=f"q{q_index}_{score}", 
                    use_container_width=True):
            handle_answer(score)
    
    # Progress bar
    progress = (q_index) / len(PHQ9_QUESTIONS)
    st.progress(progress)

def reset_assessment():
    """Reset the assessment to start over"""
    st.session_state.test_complete = False
    st.session_state.current_q = 0
    st.session_state.score = 0
    st.session_state.messages = [{"role": "assistant", "content": "Welcome back! Let's start the assessment again."}]
    st.rerun()

# ===============================================
# 4. MAIN APP EXECUTION
# ===============================================

def main():
    """The main Streamlit application function."""
    
    st.set_page_config(
        page_title="Adaptive Mental Fitness Chatbot", 
        layout="centered",
        page_icon="🧠"
    )
    st.title("🧠 MINDWELL ")
    st.markdown("---")
    
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Main Logic Flow
    if not st.session_state.test_complete:
        # Display the questionnaire until it is complete
        st.info("🔍 **Assessment in Progress** - Please answer all questions to begin chatting")
        display_phq9_ui()
        
        # Add reset button
        if st.button("Start Over", type="secondary"):
            reset_assessment()
    
    else:
        # Chatbot is active after the test
        st.success("✅ **Assessment Complete** - You can now chat with the assistant")
        
        # Add option to retake assessment
        if st.button("Retake Assessment", type="secondary"):
            reset_assessment()
        
        prompt = st.chat_input("What's on your mind today? Type 'quit' to end.")
        
        if prompt:
            # Add User Message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display User Message immediately
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process and Generate Assistant Response
            if prompt.lower().strip() in ["quit", "exit", "bye", "goodbye"]:
                response = "Thank you for trusting me today. Remember, your well-being is important. Take care of yourself."
                st.session_state.chat_active = False
            else:
                response = get_psych_response(prompt)
                
            # Display Assistant Response
            with st.chat_message("assistant"):
                st.markdown(response)

            # Add Assistant Response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Auto-rerun to update the chat display
            st.rerun()


if __name__ == '__main__':
    main()