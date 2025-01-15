import openai
from rasa_sdk import Action
from rasa_sdk.executor import CollectingDispatcher
import os

# Set your OpenAI API key (ensure it's properly set in your environment variables or replace it directly)
# Or directly use your API key here

class ActionOpenAIResponse(Action):
    def name(self) -> str:
        return "action_openai_response"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        user_message = tracker.latest_message.get('text')  # Get the user's latest message

        try:
            # Call OpenAI's API to get a response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # You can choose another model like gpt-4
                messages=[
                    {"role": "user", "content": user_message}
                ],
                max_tokens=150,
                temperature=0.7
            )

            # Extract the response text from OpenAI API
            bot_reply = response['choices'][0]['message']['content'].strip()

            # Send the response to the user
            dispatcher.utter_message(text=bot_reply)

        except openai.OpenAIError as e:
            dispatcher.utter_message(text="Sorry, I encountered an error with the OpenAI API.")
            print(f"OpenAI Error: {e}")  # Print out the OpenAI error to the console for debugging

        except Exception as e:
            dispatcher.utter_message(text="Sorry, I encountered an unexpected error.")
            print(f"Unexpected Error: {e}")  # Print out the generic error to the console for debugging

        return []
