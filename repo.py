import os
import subprocess
import sys
import time
import shutil
import webbrowser
import threading
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
import re

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Alternatively, set your API key here as a string
MAX_TOKEN_COST = 1.0  # USD
COST_PER_1000_TOKENS = 0.03  # Adjust based on actual pricing
MAX_TOKENS = int((MAX_TOKEN_COST / COST_PER_1000_TOKENS) * 1000)

# Initialize OpenAI

# Token usage tracker
class TokenCounter:
    def __init__(self):
        self.tokens_used = 0

    def add_tokens(self, tokens):
        self.tokens_used += tokens

    def has_exceeded(self):
        return self.tokens_used >= MAX_TOKENS

token_counter = TokenCounter()

def get_github_repo():
    repo_url = input("Enter the GitHub repository URL: ").strip()
    return repo_url

def clone_repo(repo_url):
    try:
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        if os.path.exists(repo_name):
            print(f"Directory '{repo_name}' already exists. Removing it.")
            shutil.rmtree(repo_name)
        subprocess.check_call(["git", "clone", repo_url])
        print(f"Cloned repository '{repo_name}'.")
        return repo_name
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        sys.exit(1)

def detect_project_type(repo_path):
    if os.path.exists(os.path.join(repo_path, 'package.json')):
        return 'node'
    elif os.path.exists(os.path.join(repo_path, 'requirements.txt')):
        return 'python'
    elif os.path.exists(os.path.join(repo_path, 'setup.py')):
        return 'python'
    else:
        return 'unknown'

def install_dependencies(repo_path, project_type):
    try:
        if project_type == 'node':
            subprocess.check_call(["npm", "install"], cwd=repo_path)
        elif project_type == 'python':
            if os.path.exists(os.path.join(repo_path, 'requirements.txt')):
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo_path)
            elif os.path.exists(os.path.join(repo_path, 'setup.py')):
                subprocess.check_call([sys.executable, "setup.py", "install"], cwd=repo_path)
        else:
            print("Unknown project type. Skipping dependency installation.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    return True

def get_error_details(error_message, repo_path, project_type):
    context = f"""
I am trying to set up a project located at {repo_path}. The project type is {project_type}.
I encountered the following error during setup:

{error_message}

Please help me troubleshoot and resolve this issue.
"""
    return context

def troubleshoot_with_gpt(error_message, repo_path, project_type):
    prompt = get_error_details(error_message, repo_path, project_type)
    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for troubleshooting software setup issues."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.5)
        tokens = response.usage.total_tokens
        token_counter.add_tokens(tokens)
        advice = response.choices[0].message.content.strip()
        return advice
    except Exception as e:
        print(f"Error communicating with OpenAI API: {e}")
        return None

def run_setup(repo_path, project_type):
    success = install_dependencies(repo_path, project_type)
    if success:
        print("Dependencies installed successfully.")
        return True
    else:
        print("Failed to install dependencies.")
        return False

def detect_run_command(repo_path, project_type):
    if project_type == 'node':
        package_json = os.path.join(repo_path, 'package.json')
        with open(package_json, 'r') as f:
            content = f.read()
            scripts = re.findall(r'"scripts":\s*{([^}]+)}', content, re.DOTALL)
            if scripts:
                scripts = scripts[0]
                if '"start"' in scripts:
                    return ["npm", "start"]
    elif project_type == 'python':
        # Example: detect if there's a Flask or Django app
        # This can be expanded based on specific requirements
        if os.path.exists(os.path.join(repo_path, 'app.py')):
            return [sys.executable, "app.py"]
    return None

def launch_app(run_command, repo_path):
    try:
        process = subprocess.Popen(run_command, cwd=repo_path)
        print(f"Started application with PID {process.pid}.")
        # Simple heuristic to find localhost URL
        time.sleep(5)  # Wait for the server to start
        ports = re.findall(r'http://localhost:(\d+)', subprocess.getoutput(f"lsof -Pan -p {process.pid} -i"))
        if ports:
            url = f"http://localhost:{ports[0]}"
            print(f"Opening {url} in web browser.")
            webbrowser.open(url)
        else:
            print("Could not detect the URL to open.")
    except Exception as e:
        print(f"Error launching application: {e}")

def main():
    repo_url = get_github_repo()
    repo_path = clone_repo(repo_url)
    project_type = detect_project_type(repo_path)
    print(f"Detected project type: {project_type}")

    setup_success = run_setup(repo_path, project_type)
    attempts = 0
    max_attempts = 5

    while not setup_success and not token_counter.has_exceeded() and attempts < max_attempts:
        attempts += 1
        print(f"Attempt {attempts} to troubleshoot and fix the setup.")
        # Assume the last error message is needed; in a real scenario, capture the actual error
        error_message = "Sample error message for troubleshooting."  # Placeholder
        advice = troubleshoot_with_gpt(error_message, repo_path, project_type)
        if advice:
            print("GPT-4 suggested the following steps to fix the issue:")
            print(advice)
            # Here you might implement applying the advice automatically
            # For simplicity, we'll stop and ask the user to take action
            print("Please apply the suggested fixes and try running the setup again.")
            input("Press Enter after you've applied the fixes...")
            setup_success = run_setup(repo_path, project_type)
        else:
            print("No advice received from GPT-4.")
            break

    if not setup_success:
        print("\n--- Setup Summary ---")
        print(f"Repository: {repo_url}")
        print(f"Project Type: {project_type}")
        print(f"Token Usage: {token_counter.tokens_used}")
        print("Setup was not completed successfully. Please review the above messages and troubleshoot further.")
        sys.exit(1)

    # If setup is successful, check if the project should run a local server/app
    run_command = detect_run_command(repo_path, project_type)
    if run_command:
        launch_app(run_command, repo_path)
    else:
        print("No run command detected. Setup is complete.")

if __name__ == "__main__":
    main()
