from google import genai
from json_repair import repair_json
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

    def llm_chat(self,model: str = "gemini-3-flash-preview", context=None):
        prompt = f"""You are an assistant that generates city recommendations based on a person's travel preferences, style, and lifestyle.
        
        Context: {context}
        Your Task is to generate a list of cities suitable for this person based on the above context. Organize the recommendations into the following JSON structure:
        {{
            "cities": [ /* list a city suitable with countries for this user in IATA code*/ ],
            "city_names":[ /* name of the city],
            "city_description": [ /* Corresponding city description]
            "recommended_duration_days": [ /* suggested number of days to spend in each city */ ],
            "origin": [ /* where the person starts journey from in IATA code or the nearest famous airport code to the origin],
            "currency":[ /* The local currency which the user lives in and uses]
        }}
        
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

        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents = prompt
            )

            if response.text:
                return response.text.strip()
            else:
                data = repair_json(response.text.strip(),return_objects=True)
                return data

        except Exception as e:
            print(f"Error:{str(e)}")
            return None

    def fetch_destination(self):
        query_embeds=create_embeddings("Describe the user's travel preferences, including favorite destinations, travel style (adventurous, relaxed, cultural, etc.), budget range, accommodation and transport choices, preferred activities, and travel companions. Include typical trip length, planning style, favorite climates or seasons, and how they express themselves through travel (e.g., photography, local immersion, sustainability). Mention any travel-related hobbies, booking habits, or loyalty programs.")
        if query_embeds:
            try:
                results = vectors.search(query=query_embeds).to_list()[:5]
                