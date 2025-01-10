import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "m42-health/Llama3-Med42-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the inference function
def medical_response(query):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=200, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create a Gradio interface
interface = gr.Interface(
    fn=medical_response,
    inputs=gr.Textbox(lines=4, placeholder="Enter your medical question here..."),
    outputs=gr.Textbox(label="Model Response"),
    title="Llama3-Med42-8B Medical Assistant",
    description="Ask any medical-related questions to the Llama3-Med42-8B model.",
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
