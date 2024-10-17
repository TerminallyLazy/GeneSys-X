# Standard Library
import os
from io import StringIO
from time import time
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

logging.info("Loading External Modules")
# External Modules
import gradio as gr
import pandas as pd
from pandasai import SmartDataframe
from pandasai.schemas.df_config import Config

logging.info("Loading Internal Modules")
# Internal Modules
from genesys.env import load_dotenv
from genesys.visuals import render_protein_file, create_protein_interface
from genesys.ai import run_conversation
import genesys.client as cli

load_dotenv()

logging.info("Loading Temp Directory")
# Different ways of handling local temp directory.
if os.name == 'nt':  # Windows
    temp_dir = os.getenv('TEMP')
else:  # UNIX-like OS (Mac & Linux)
    temp_dir = "/tmp"

logging.info("Determining File Type")
def determine_file_type(file):
    if file is not None:
        file_extension = file.name.split('.')[-1].lower()
        if file_extension == "fasta":
            return "FASTA"
        elif file_extension == "csv":
            return "CSV"
        elif file_extension == "pdb":
            return "PDB"
    return None

def process_fasta(file, username):
    try:
        logging.info(f"Starting process_fasta with file: {file}, username: {username}")
        if hasattr(file, 'name'):
            filename = f"{str(int(time()))}-{file.name}"
            logging.info(f"Generated filename: {filename}")
        else:
            filename = f"{str(int(time()))}-uploaded_fasta.fasta"
            logging.info(f"Generated default filename: {filename}")

        if hasattr(file, 'read'):
            logging.info("File has 'read' attribute, reading content")
            fasta_content = file.read()
            if isinstance(fasta_content, bytes):
                logging.info("Content is in bytes, decoding to utf-8")
                fasta_content = fasta_content.decode('utf-8')
        else:
            logging.info("File doesn't have 'read' attribute, assuming it's already content")
            fasta_content = file  # Assume it's already the file content

        logging.info(f"Uploading content to S3 with username: {username}, filename: {filename}")
        upload_result = cli.upload_s3(fasta_content, username, filename, "FASTA")
        logging.info(f"Upload result: {upload_result}")
        return filename, "FASTA file processed successfully"
    except IOError as e:
        logging.error(f"Error reading FASTA file: {str(e)}")
        return None, f"Error processing FASTA file: {str(e)}"
    except UnicodeDecodeError as e:
        logging.error(f"Error decoding FASTA file content: {str(e)}")
        return None, f"Error decoding FASTA file: {str(e)}"
    except Exception as e:
        logging.error(f"Unexpected error processing FASTA file: {str(e)}", exc_info=True)
        return None, f"Unexpected error processing FASTA file: {str(e)}"

def process_csv(file, username):
    df = pd.read_csv(file)
    csv_filename = f"{str(int(time()))}-{file.name}"
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    cli.upload_s3(csv_buffer.getvalue(), username, csv_filename, "csv")
    return df, "CSV file processed successfully"

def process_pdb(file, username):
    pdb_content = file.read().decode("utf-8")
    pdb_filename = f"{str(int(time()))}-{file.name}"
    cli.upload_s3(pdb_content, username, pdb_filename, "pdb")
    return pdb_content, "PDB file processed successfully"

def process_file(file, username):
    file_type = determine_file_type(file)
    if file_type == "FASTA":
        return process_fasta(file, username)
    elif file_type == "CSV":
        return process_csv(file, username)
    elif file_type == "PDB":
        return process_pdb(file, username)
    else:
        return None, "Unsupported file type"

def answer_question(file, question, username):
    file_type = determine_file_type(file)
    if file_type == "FASTA":
        temp_file_path, _ = process_fasta(file, username)
        return run_conversation(question, temp_file_path)
    elif file_type == "CSV":
        df, _ = process_csv(file, username)
        sdf = SmartDataframe(df, config=Config())
        return sdf.chat(question)
    elif file_type == "PDB":
        pdb_content, _ = process_pdb(file, username)
        return render_protein_file(pdb_content)
    else:
        return "Unsupported file type for question answering"

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🧬 GeneSys AI 🧬")
        gr.Markdown("*Making it as easy as AUG*")

        username = gr.Textbox(label="Username", placeholder="Enter your username")

        with gr.Tab("File Processing"):
            with gr.Row():
                file_input = gr.File(label="Upload FASTA, CSV, or PDB file")
                file_type_output = gr.Textbox(label="Detected File Type")

            process_button = gr.Button("Process File")
            process_output = gr.Textbox(label="Processing Result")

            with gr.Row():
                question_input = gr.Textbox(label="Ask a question about your data")
                answer_button = gr.Button("Get Answer")

            answer_output = gr.Textbox(label="Answer")

        with gr.Tab("Protein Visualization"):
            protein_interface = create_protein_interface()

        def update_file_type(file):
            return determine_file_type(file)

        def process_uploaded_file(file, username):
            logging.info(f"Processing file: {file.name}")
            logging.info(f"Username: {username}")
            result, message = process_file(file, username)
            logging.info(f"Process result: {result}")
            logging.info(f"Process message: {message}")
            return message

        def answer_user_question(file, question, username):
            return answer_question(file, question, username)

        file_input.change(update_file_type, inputs=[file_input], outputs=[file_type_output])
        process_button.click(process_uploaded_file, inputs=[file_input, username], outputs=[process_output])
        answer_button.click(answer_user_question, inputs=[file_input, question_input, username], outputs=[answer_output])

    return demo

if __name__ == "__main__":
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        logging.info("Create Session for Event Creator")
        unix_time = str(int(time()))
        session_id = f"session-{unix_time}"

        demo = create_interface()
        demo.launch(share=True)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
