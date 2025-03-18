from flask import Flask, render_template, request, redirect, url_for, session

import os
import openai
from setup_crewai_agents import QuestionSetter, TeachingAssistant
from setup_llamaindex import LlamaIndexQueryEngine
from datetime import datetime
import ast
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)

QA_HISTORY_FILE = "qa_history.txt"
GEN_QUESTION_HISTORY_FILE = "gen_question_history.txt"

open_ai_key = os.getenv('OPENAI_API_KEY')
if not open_ai_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


# initialize the CrewAI agents
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
PAST_QUESTION_FILE = os.getenv('PAST_QUESTION_FILE')
MAIN_MATERIAL_FILE = os.getenv('MAIN_MATERIAL_FILE')
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = data_dir=os.path.join(current_dir, "../data")
ta = TeachingAssistant(data_dir=data_dir, llm_model=LLM_MODEL)
qs = QuestionSetter(question_example_file=PAST_QUESTION_FILE, main_material_file=MAIN_MATERIAL_FILE)
print("[INFO] Study Assistant initialized")

# initialize LlamaIndex
llama_engine = LlamaIndexQueryEngine(data_dir=data_dir, llm_model=LLM_MODEL)
print("[INFO] LlamaIndex initialized")


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Gen question
        if "gen_question" in request.form:
            print("[INFO] gen_question clicked")
            try:
                qs_result = qs.crew.kickoff()
                print("Result:\n", qs_result.raw)
                # html_content = markdown.markdown(qs_result.raw)
                html_content = qs_result.raw
                if html_content.startswith("```html"):
                    html_content = html_content[8:-3]
            except Exception as e:
                print(f"[ERROR] Error during kickoff: {e}")
                return render_template('index.html', examples="Error occurred during processing. Please try again.")
            
            # add to history
            save_history(GEN_QUESTION_HISTORY_FILE, (request_time, html_content))
            return render_template('index.html', examples=html_content)
        
        # Answer question
        if "answer_question" in request.form:
            print("[INFO] answer_question clicked")
            user_prompt = request.form.get('text_input')
            print(f"[INFO] user_prompt is provided: {user_prompt}")
            elected_option = request.form.get("options")
            print(f"[INFO] mode is {elected_option}")

            if elected_option == "crewai":
                try:
                    ta_result = ta.crew.kickoff(inputs={"question": user_prompt})
                    html_content = ta_result.raw
                    if html_content.startswith("```html"):
                        html_content = html_content[8:-3]
                except Exception as e:
                    print(f"[ERROR] Error during kickoff: {e}")
                    return render_template('index.html', query=user_prompt, answer=f"Error occurred during processing. {e} <br>Please try again.")

                # add to history
                save_history(QA_HISTORY_FILE, (request_time, user_prompt, elected_option, html_content))
                return render_template('index.html', query=user_prompt, answer=html_content)
            
            elif elected_option == "llama":
                try:
                    if '“' in user_prompt:
                        user_prompt = user_prompt.replace('“', "'")
                    if '"' in user_prompt:
                        user_prompt = user_prompt.replace('"', "'")
                    response = llama_engine.query_engine.query(user_prompt)
                    print(f"[INFO] response: {response}")
                except openai.RateLimitError as e:
                    error_message = f"Rate limit exceeded: {e}"
                    return render_template('index.html', query=user_prompt, answer=f"Error occurred during processing. {error_message} <br>Please try again.")
                except Exception as e:
                    print(f"[ERROR] Error during query: {e}")
                    return render_template('index.html', query=user_prompt, answer=f"Error occurred during processing. <br>Please try again.")
                save_history(QA_HISTORY_FILE, (request_time, user_prompt, elected_option, response.response))
                return render_template('index.html', query=user_prompt, answer=response)
    return render_template('index.html')


@app.route("/history")
def history():
    history_gen_question = load_history(GEN_QUESTION_HISTORY_FILE)
    print("history_gen_question", history_gen_question)
    history_qa = load_history(QA_HISTORY_FILE)
    return render_template("history.html", history_gen_question=history_gen_question, history_qa=history_qa)


def load_history(history_file):
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            return [ast.literal_eval(line.strip()) for line in f.readlines()]
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

def save_history(history_file, data: tuple):
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(str(data) + "\n")

if __name__ == '__main__':
    app.run(debug=True)


