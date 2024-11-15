from flask import Flask, request, jsonify
import openai

# Replace with your OpenAI API key
openai.api_key = "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"

# Initialize Flask app
app = Flask(_name_)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    vehicle_data = request.json.get('vehicle_data')  # Optional, you can pass vehicle data for personalization

    # Create the prompt for the chatbot, including vehicle data if available
    prompt = f"User: {user_input}\nBot:"
    if vehicle_data:
        prompt = f"Vehicle: {vehicle_data}\n{prompt}"

    # Get response from OpenAI's GPT model
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can use other engines like GPT-4 if available
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6
    )

    # Extract the response text
    answer = response.choices[0].text.strip()
    return jsonify({"response": answer})

if _name_ == "_main_":
    app.run(port=5000, debug=True)