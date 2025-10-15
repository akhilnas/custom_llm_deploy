# inference.py

import os
import json
import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# =====================================================================================
# 1. model_fn
#    - This function is called ONCE when the SageMaker endpoint is created.
#    - It is responsible for loading the model and processor from the /opt/ml/model
#      directory (where SageMaker mounts your model files) into memory.
#    - The returned object (in this case, a dictionary) is then passed to the
#      predict_fn on every invocation.
# =====================================================================================
def model_fn(model_dir):
    """
    Load the model and processor from the disk.
    """
    print(f"Loading model from directory: {model_dir}...")
    
    # Check if a GPU is available and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Use 4-bit quantization to reduce memory footprint and improve performance
    # bnb_config stands for "bitsandbytes config"
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the model from the directory with quantization enabled.
    # trust_remote_code=True is required for Gemma 3.
    # device_map="auto" lets the library automatically place model parts on available GPUs.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load the associated processor, which is crucial for preparing
    # both the text and image data for the model.
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)

    print("Model and processor loaded successfully.")
    
    # Return the loaded artifacts as a dictionary.
    # This will be passed to predict_fn.
    return {"model": model, "processor": processor, "device": device}


# =====================================================================================
# 2. input_fn
#    - This function is called for EVERY invocation of the endpoint.
#    - It is responsible for deserializing the incoming request payload.
#    - It should parse the data and return it in a format that predict_fn can use.
# =====================================================================================
def input_fn(request_body, request_content_type):
    """
    Parse the incoming request. Expects a JSON object with 'text' and 'image' keys.
    The 'image' is expected to be a base64 encoded string.
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)

        # Extract text prompt and base64 encoded image from the request
        text_prompt = data.get('text')
        image_b64 = data.get('image')

        if not text_prompt or not image_b64:
            raise ValueError("Request must be a JSON object with 'text' and 'image' keys.")

        # Decode the base64 string back into image bytes
        try:
            image_bytes = base64.b64decode(image_b64)
            # Open the image using Pillow (PIL)
            image = Image.open(BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Failed to decode or open the image. Error: {e}")

        # Return the processed inputs ready for the predict_fn
        return {"prompt": text_prompt, "image": image}
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}. Must be 'application/json'.")


# =====================================================================================
# 3. predict_fn
#    - This function is called for EVERY invocation.
#    - It takes the model artifacts from model_fn and the processed input from input_fn.
#    - This is where the actual inference (prediction) happens.
# =====================================================================================
def predict_fn(input_data, model_artifacts):
    """
    Generate text from the model based on the text and image inputs.
    """
    print("Received request for prediction...")
    
    # Unpack model artifacts and input data
    model = model_artifacts['model']
    processor = model_artifacts['processor']
    device = model_artifacts['device']
    
    prompt = input_data['prompt']
    image = input_data['image']
    
    # This is the critical multimodal step. The processor takes the raw text and
    # the PIL image and correctly formats them into tensors that the model expects.
    # The chat template ensures the input is structured in a conversational format.
    chat = [{"role": "user", "content": [f"<image>\n{prompt}"]}]
    prompt_for_model = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(text=prompt_for_model, images=image, return_tensors="pt").to(device)

    print("Generating output...")
    # Generate output using the model.
    # We use a context manager to ensure no gradients are calculated, which saves memory.
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024) # Adjust max_new_tokens as needed

    # The output from the model is a tensor of token IDs. The processor is used
    # again to decode these IDs back into human-readable text.
    generated_ids = output[:, inputs['input_ids'].shape[1]:] # Get only the generated tokens
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print("Prediction complete.")
    return generated_text


# =====================================================================================
# 4. output_fn
#    - This function is called for EVERY invocation.
#    - It takes the prediction result from predict_fn and serializes it into the
#      final response format that will be sent back to the client.
# =====================================================================================
def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result into the desired JSON response format.
    """
    if response_content_type == 'application/json':
        # Structure the final response
        response = {"generated_text": prediction}
        return json.dumps(response)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}. Must be 'application/json'.")