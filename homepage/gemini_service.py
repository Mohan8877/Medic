from google import genai
from google.genai import types

# 1. Configuration
API_KEY = "AIzaSyDBNYcboD2GweS3KM8woo8pchoo-2fndQY" 
client = genai.Client(api_key=API_KEY)

def get_self_care_advice(disease_name, language="English"):
    """
    Takes a disease name and returns structured self-care advice using the new SDK.
    """
    try:
        # 2. Use the latest model (Gemini 3.1 Flash)
        # 3. Define the System Instruction (The Persona)
        config = types.GenerateContentConfig(
            system_instruction="""
ROLE: You are a senior Clinical Triage Officer working in a rural medical camp. 
TONE: Professional, direct, and clinical. Avoid conversational filler, empathetic apologies, or "AI-sounding" intros/outros.
STYLE: Use the "Active Voice." Replace complex medical jargon with simplified terms that a rural patient can act upon.
CONSTRAINTS: 
1. DO NOT use words like "leverage," "furthermore," "delve," or "important to note."
2. DO NOT offer a definitive diagnosis. Use terms like "Symptoms are consistent with..." or "Suspected case of..."
3. DO NOT include flowery introductions like "I'm here to help." Start immediately with the medical content.
4. STRUCTURE: Use Markdown headers and clean bullet points for readability on small mobile screens.
""",
            temperature=0.2,
            max_output_tokens=500,
        )

        prompt = f"""
        Respond in {language}.
        PATIENT CASE: Suspected {disease_name}
INSTRUCTION: Provide a professional medical advisory for this condition.

OUTPUT FORMAT:
## Clinical Summary
[A 1-sentence description of the condition in simple terms]

## Immediate Actions
- [Action 1]
- [Action 2]
- [Action 3]

## Nutrition & Hydration
- [Specific dietary advice for this condition]

## RED FLAGS 
# Seek Emergency Care if you have
- [List 3 critical symptoms that require immediate hospital visit]

## Disclaimer
NOT A REPLACEMENT FOR IN-PERSON MEDICAL CONSULTATION.
        """

        # 4. Generate the response
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview", # Fast and free for hackathons
            contents=prompt,
            config=config
        )
        
        return response.text

    except Exception as e:
        return f"Error fetching advice: {str(e)}"


def get_fallback_diagnosis(user_text):
    """
    Called when the local ML model is inconclusive. 
    Gemini analyzes the raw text to provide a likely condition and risk level.
    """
    try:
        config = types.GenerateContentConfig(
            system_instruction="You are a clinical diagnostic assistant. Analyze the user's symptoms and provide a likely condition. Be concise.",
            temperature=0.1,
        )

        prompt = f"""
        User Symptoms: {user_text}
        
        Based on these symptoms, provide:
        1. Likely Condition
        2. Risk Level (Low, Moderate, High, or Critical)
        3. A 0-100 Confidence Score
        
        OUTPUT ONLY IN THIS JSON FORMAT:
        {{"diagnosis": "Condition Name", "level": "level_name", "score": 85}}
        """

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=config
        )
        
        # Parse the JSON from Gemini's response
        import json
        return json.loads(response.text)
    except Exception as e:
        return {"diagnosis": "Inconclusive", "level": "moderate", "score": 0}
    
def get_hybrid_diagnosis(user_text, local_prediction=None):
    """
    Acts as an expert cross-checker. Analyzes symptoms and compares them 
    against the local model's output to provide a perfect result.
    """
    try:
        config = types.GenerateContentConfig(
            system_instruction="""
            ROLE: Senior Medical Consultant.
            TASK: Diagnose symptoms provided by rural patients.
            TONE: Professional, concise, and definitive.
            FORMAT: JSON only.
            """,
            temperature=0.1, # Low temperature for consistent medical facts
        )

        prompt = f"""
        PATIENT INPUT: "{user_text}"
        LOCAL MODEL SUGGESTION: "{local_prediction if local_prediction else 'None'}"

        INSTRUCTIONS:
        - If the local model suggestion is 'None' or looks wrong for the symptoms, provide the correct medical diagnosis.
        - Determine Risk Level: Low, Moderate, High, or Critical.
        - Provide a 0-100 Confidence score.

        REQUIRED JSON OUTPUT:
        {{
          "diagnosis": "Condition Name",
          "level": "moderate",
          "score": 85,
          "reasoning": "Brief explanation of why this was chosen."
        }}
        """

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=config
        )
        
        import json
        return json.loads(response.text)
    except Exception as e:
        return {"diagnosis": "System Consultation Required", "level": "moderate", "score": 0}