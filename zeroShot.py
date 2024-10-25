import openai
import os
import time  # Added import for time module

# Initialize the OpenAI API key
openai.api_key = ""
statements = [
    "1. Supply and demand dynamics significantly impact food inflation rates. Inelastic demand for staple foods leads to larger price increases when supply decreases.",
    "2. Seasonal variations in supply affect the price elasticity of demand for fruits and vegetables.",
    "3. The impact of food price changes on consumer purchasing behavior is stronger in lower-income households.",
    "4. Production costs are a major driver of food inflation. Economies of scale in food production lead to lower prices and less inflation in larger markets.",
    "5. Market structure influences food inflation. Inflation expectations shape consumer behavior and pricing strategies. Anticipated inflation leads consumers to increase current food purchases, driving prices higher.",
    "6. Price-setting behavior among retailers reflects inflation expectations, influencing food inflation rates."
]

# Function to analyze transcripts in a directory
def analyze_transcripts(directory):
    transcripts = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                transcripts[filename] = content
    return transcripts

# Function to generate the prompt for GPT
def generate_prompt(statement, transcript):
    return f"""
Assign a score from 1 to 5 for the following statement based on its importance according to the context:
Statement: "{statement}"
Transcript: {transcript}
Scoring Scale:
1 = Not Important
2 = Slightly Important
3 = Moderately Important
4 = Very Important
5 = Extremely Important

Provide the score only as an integer.
"""

# Function to get the score using GPT with temperature and max_tokens settings
def get_score(statement, transcript, temperature=0.9, max_tokens=1000):
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Using GPT-4o model
        messages=[{"role": "user", "content": generate_prompt(statement, transcript)}],
        temperature=temperature,  # Control creativity
        max_tokens=max_tokens  # Limit the length of the response
    )
    score = response.choices[0]['message']['content'].strip()
    time.sleep(5)  # Add a buffer period of 5 seconds
    return score

# Directory containing transcript files
transcripts_directory = "/Users/tirthoroy/Downloads/FoodInflationAI/dataset"  # Set the path to your transcripts directory
transcripts = analyze_transcripts(transcripts_directory)

# Prepare to write scores to a text file
with open('zeroShotScores.txt', mode='w') as txt_file:
    # Iterate over each transcript, get scores for each statement, and write to TXT
    for filename, transcript in transcripts.items():
        txt_file.write(f"{filename}\n")  # Write the transcript name to the text file
        for statement in statements:
            score = get_score(statement, transcript, temperature=0.5, max_tokens=20)  # Get score for the statement
            txt_file.write(f"Score: {score}\n")  # Write the statement and score to the text file

        # Add a separator for clarity in the text file
        txt_file.write("\n" + "-" * 50 + "\n")

print("Scoring completed and saved to 'food_inflation_scores.txt'")
