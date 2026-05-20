from google import genai
from json import json
from json_repair import repair_json
from datetime import datetime
import lancedb
from dotenv import load_dotenv

db_path = base_path / "vectordb"
collection = lancedb.connect(db_path)
vectors = collection.open_table("vectors")
load_dotenv()

client = genai.Client(api_key='GEMINI_API_KEY')
class TravelPlanner():
    def __init__(self, api_key, client):
        self.api_key = api_key
        self.client = client
        self.vectors = vectors
        self.destinations = None
        self.start_date = None
        self.end_date = None

    def LLM_Chat(self,model: str = "gemini-3-flash-preview", context=None):
        json_schema = {
            "type": "OBJECT",
            "properties": {
                "cities": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Exactly one destination city 3-letter IATA code."},
                "city_names": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "The full common name of the destination city."},
                "city_description": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "A description of why this city fits the user."},
                "start_date": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "The Starting date of the trip.Usually the next day of search if no input is mentioned"},
                "end_date": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "The Ending date of the trip."},
                "origin": {"type": "STRING", "description": "The 3-letter IATA code of the user's starting airport."},
                "currency": {"type": "STRING", "description": "The local currency code used where the user lives."}
            },
            "required": ["cities", "city_names", "city_description", "start_date", "end_date", "origin", "currency"]
        }

        instructions = f"""You are an assistant that generates city recommendations based on a person's travel preferences, style, and lifestyle.
    
        Your Task is to generate a list of cities suitable for this person based on the above context. Organize the recommendations into the following JSON structure:
        
        Requirements:
        - Output valid JSON only; do not include extra commentary.
        - The "origin" IATA code must never match any city in "cities", ensuring each listed destination is distinct
        - Include cities that match the user's preferred travel style, activities, climate, and pace.
        - Prioritize cities that are versatile for leisure, culture, heritage, adventure, or relaxation depending on the context.
        - Include a mix of popular and off-the-beaten-path destinations if suitable.
        - Ensure the number of cities does not exceed 1.
        - Make sure all cities are in the same country
        - Provide all the places in IATA codes used in Airports
        - Make sure origin and cities don't match
        - Make sure the output is in the given format exactly and no fields are missed
        - Output **only** the data in this structure. Do not include explanations, commentary, or any mention of JSON or code blocks.
        """

        llm_config = {
            "system_instruction": instructions,
            "response_mime_type": "application/json",
            "response_schema": json_schema
        }

        try:
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents = [context],
                config=llm_config
            )
        except Exception as api_error:
            print(f"Gemini API Call failed: {str(api_error)}")
            return None

        if response and response.text:
            try:
                return json.loads(response.text.strip())
            except Exception as e:
                print(f"Error:{str(e)}")
                try:
                    return repair_json(response.text.strip(),return_objects=True)
                except Exception:
                    return None   
        return None

    def fetch_destination(self, user_input):
        query = (f"Take this user_prompt: {user_input} and Describe the user's travel preferences, including favorite destinations, travel style (adventurous, relaxed, cultural, etc.), budget range, accommodation and transport choices, preferred activities, and travel companions. Include typical trip length, planning style, favorite climates or seasons, and how they express themselves through travel (e.g., photography, local immersion, sustainability). Mention any travel-related hobbies, booking habits, or loyalty programs.")
        query_embeds=create_embeddings(query)
        if query_embeds:
            try:
                results = vectors.search(query=query_embeds).to_list()[:5]
                context = " ".join([(r["text"]) for r in results])
                json_output = self.llm_chat(context=context)
                self.destinations = json.loads(json_output)["city_names"][0]
                return self.destinations
            except Exception as e:
                print(f"Failed to Fetch destination due to the following error {str(e)}")
                return None

    def fetch_dates(self,user_prompt)
        query = (f"Take this user_prompt: {user_input} and Extract the precise timeline, calendar dates, and duration for the trip described in the user_input. Identify the explicit start date and the number of days of the trip to calculate or locate the exact end date based on the total trip length/number of days per destination, or returning dates mentioned. Focus on temporal keywords like days, weeks, months, specific dates associated with this travel plan.")
        query_embeds=create_embeddings(query)
        if query_embeds:
            try:
                results = vectors.search(query=query_embeds).to_list()[:5]
                context = " ".join([(r["text"]) for r in results])
                json_output = self.llm_chat(context=context)
                self.start_date = json.loads(json_output)["start_date"][0]
                self.end_date = json.loads(json_output)["end_date"][0]
                return [self.start_date,self.end_date]
            except Exception as e:
                print(f"Failed to Fetch date due to the following error {str(e)}")
                return None

    def fetch_weather(self,city,start_date,end_date):
        base_url = f"https://serpapi.com/search.json?q=whats+the+weather+condition+in+the+{city}+from+{start_date}+to+{end_date}&location=United+States&hl=en&gl=us&google_domain=google.com"
        params = {api_key=Weather_API}
        response = requests.get(base_url,params=params)
        if response.status_code == 200:
            data = response.json()["answer_box"]["forecast"]
        print(response["forecast"][])