import os
from openai import OpenAI
import PyPDF2
from fpdf import FPDF
import tiktoken
from tqdm import tqdm  # For progress bar

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # Ensure your API key is stored as an environment variable
)

# Get the correct encoding for GPT-4o-mini
def get_gpt4o_mini_encoding():
    print("Getting GPT-4o-mini encoding...")
    model_name = "gpt-4o-mini"
    encoding = tiktoken.encoding_for_model(model_name)
    return encoding

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    pdf_text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text += page.extract_text()
    print("Text extraction complete.")
    return pdf_text

# Function to split text into smaller chunks based on the 15,000 token limit
def split_text_into_chunks(text, encoding, max_input_tokens=15000):
    """Split the text into smaller chunks that fit within the 15,000 input token limit."""
    print("Splitting text into chunks...")
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_input_tokens):
        chunk_tokens = tokens[i:i + max_input_tokens]
        chunks.append(encoding.decode(chunk_tokens))
    
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

# Function to translate a chunk using GPT-4o-mini
def translate_chunk(chunk, target_language):
    print(f"Translating chunk (size: {len(chunk)} characters)...")
    input_messages = [
        {"role": "system", "content": f"You are a helpful assistant that translates text to {target_language}."},
        {"role": "user", "content": chunk}
    ]
    
    # Get the encoding for the model
    encoding = get_gpt4o_mini_encoding()

    # Calculate input token count
    input_token_count = sum(len(encoding.encode(message['content'])) for message in input_messages)
    
    # Set a limit for the completion tokens to be at most 16,384 and fit within the overall 127,000 token limit
    max_completion_tokens = min(16384, 127000 - input_token_count)

    try:
        # Make the API call with the adjusted max tokens for completion
        chat_completion = client.chat.completions.create(
            messages=input_messages,
            model="gpt-4o-mini",
            max_tokens=max_completion_tokens
        )
        print(f"Chunk translated successfully.")
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error occurred during translation: {e}")
        return ""

# Function to save translated text to a new PDF
def save_text_to_pdf(text, output_pdf_path):
    print(f"Saving translated text to {output_pdf_path}...")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(output_pdf_path)
    print(f"PDF saved successfully.")

# Main function to extract text, translate it in chunks, and save as a new PDF
def translate_pdf(pdf_path, output_pdf_path, target_language):
    print(f"Starting translation of {pdf_path} to {target_language}...")

    # Step 1: Extract text from the original PDF
    original_text = extract_text_from_pdf(pdf_path)

    # Step 2: Get the model encoding and split text into chunks that fit within the 15,000 input token limit
    encoding = get_gpt4o_mini_encoding()
    chunks = split_text_into_chunks(original_text, encoding, max_input_tokens=15000)

    # Step 3: Translate each chunk with a progress bar
    translated_chunks = []
    with tqdm(total=len(chunks), desc="Translating") as pbar:
        for i, chunk in enumerate(chunks):
            translated_chunk = translate_chunk(chunk, target_language)
            translated_chunks.append(translated_chunk)
            pbar.update(1)

    translated_text = "\n\n".join(translated_chunks)

    # Step 4: Save the translated text to a new PDF
    save_text_to_pdf(translated_text, output_pdf_path)

    print(f"Translated PDF saved to {output_pdf_path}")

# Example usage
if __name__ == "__main__":
    input_pdf_path = "./book.pdf"
    output_pdf_path = "translated_pdf.pdf"
    target_language = "Turkish"  # or any other language

    translate_pdf(input_pdf_path, output_pdf_path, target_language)
