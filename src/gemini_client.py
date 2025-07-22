
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

    def filter_news(self, headlines):
        
        if not headlines:
            return []

        
        prompt = f"""
        Analyze the following headlines and classify them as "core finance" or "non-core finance".
        Return a JSON object with a single key "classifications" which is a list of strings, where each string is either "core finance" or "non-core finance".

        Example:
        Headlines:
        - "Google stock price soars after strong earnings report"
        - "New movie release breaks box office records"

        Output:
        {{
            "classifications": [
                "core finance",
                "non-core finance"
            ]
        }}

        Headlines:
        {headlines}
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return None
