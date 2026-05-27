import lancedb
import json
import os
import requests
import requests_cache
import openmeteo_requests
from retry_requests import retry
from google import genai
from json_repair import repair_json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

base_path = Path(__file__).resolve().parent
db_path = base_path / "vectordb"
collection = lancedb.connect(str(db_path))

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

initial_data = [{
    "vector": [0.0] * 3072,
    "text": "initial placeholder text"
}]

vectors = collection.create_table("vectors", data=initial_data, mode="overwrite")

class TravelPlanner():
    def __init__(self, api_key: str, client) -> None:
        self.api_key = api_key
        self.client = client
        self.vectors = vectors
        self.destinations = None
        self.start_date = None
        self.end_date = None

    def LLM_Chat(self,model: str = "gemini-2.5-flash", context=None):
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
                model="gemini-2.5-flash",
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

    def fetch_destination(self, user_prompt):
        query = (f"Take this user_prompt: {user_prompt} and Describe the user's travel preferences, including favorite destinations, travel style (adventurous, relaxed, cultural, etc.), budget range, accommodation and transport choices, preferred activities, and travel companions. Include typical trip length, planning style, favorite climates or seasons, and how they express themselves through travel (e.g., photography, local immersion, sustainability). Mention any travel-related hobbies, booking habits, or loyalty programs.")
        self.query_embeds=self.create_embeddings(query)
        if self.query_embeds:
            try:
                results = vectors.search(query=self.query_embeds).to_list()[:5]
                context = " ".join([(r["text"]) for r in results])
                today_str = datetime.now().strftime('%Y-%m-%d')
                full_context = (
                    f"INSTRUCTION: Today's date is strictly {today_str}. "
                    f"Calculate any relative terms like 'today', 'tomorrow', or 'next week' using this date anchor.\n\n"
                    f"Historical Context:\n{context}\n\n"
                    f"Current Request:\n{user_prompt}"
                )
                json_output = self.LLM_Chat(context=full_context)
                self.destinations = json_output["city_names"][0]
                return self.destinations
            except Exception as e:
                print(f"Failed to Fetch destination due to the following error {str(e)}")
                return None

    def fetch_dates(self,user_prompt):
        query = (f"Take this user_prompt: {user_prompt} and Extract the precise timeline, calendar dates, and duration for the trip described in the user_input as a string in the format '%Y-%m-%d' . Identify the explicit start date and the number of days of the trip to calculate or locate the exact end date based on the total trip length/number of days per destination, or returning dates mentioned. Focus on temporal keywords like days, weeks, months, specific dates associated with this travel plan.")
        self.query_embeds=self.create_embeddings(query)
        if self.query_embeds:
            try:
                results = self.vectors.search(query=self.query_embeds).to_list()[:5]
                context = " ".join([(r["text"]) for r in results])
                today_str = datetime.now().strftime('%Y-%m-%d')
                full_context = (
                    f"INSTRUCTION: Today's date is strictly {today_str}. "
                    f"Calculate any relative terms like 'today', 'tomorrow', or 'next week' using this date anchor.\n\n"
                    f"Historical Context:\n{context}\n\n"
                    f"Current Request:\n{user_prompt}"
                )
                json_output = self.LLM_Chat(context=full_context)
                self.start_date = json_output["start_date"][0]
                self.end_date = json_output["end_date"][0]
                dates = [self.start_date,self.end_date]
                return dates
            except Exception as e:
                print(f"Failed to Fetch date due to the following error {str(e)}")
                return None

    def fetch_weather(self,city,start_date,end_date):
        # start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        # start_day = start_dt.strftime('%A')
        # end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        # end_day = end_dt.strftime('%A')
        city = city.strip()
        base_url_location =f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
        response = requests.get(base_url_location)
        data = response.json()
        latitude = data["results"][0]["latitude"]
        longitude = data["results"][0]["longitude"]
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_probability_max", "precipitation_hours"],
            "timezone": "auto",
            "start_date": start_date,
            "end_date": end_date,
        }
        base_url_weather = "https://api.open-meteo.com/v1/forecast"
        responses = openmeteo.weather_api(base_url_weather, params = params)
        response = responses[0]
        daily = response.Daily()
        daily_weather_code = daily.Variables(0).ValuesAsNumpy().tolist()
        daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy().tolist()
        daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy().tolist()
        daily_precipitation_probability_max = daily.Variables(3).ValuesAsNumpy().tolist()
        daily_precipitation_hours = daily.Variables(4).ValuesAsNumpy().tolist()
        return [daily_weather_code,daily_temperature_2m_max,daily_temperature_2m_min,daily_precipitation_probability_max,daily_precipitation_hours]

    def create_embeddings(self, text, model="models/text-embedding-004"):
        text = text.replace("\n", " ")
        result = self.client.models.embed_content(
            model="gemini-embedding-2",
            contents=text)
        return result.embeddings[0].values

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    planner = TravelPlanner(api_key=api_key,client=client)
    user_input = input("Enter your prompt:")
    destination = planner.fetch_destination(user_input)
    dates = planner.fetch_dates(user_input)
    if destination and dates:
        print(f"Destination found: {destination}")
        print(f"Dates found: {dates}")
        weather_data = planner.fetch_weather(destination,dates[0],dates[1])
        print(weather_data)

if __name__ == "__main__":
    main()