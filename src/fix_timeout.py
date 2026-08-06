with open("main.py", "r") as f:
    content = f.read()

old = 'client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))'
new = '''client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=20000),  # 20 seconds, in milliseconds
)'''

if old in content:
    content = content.replace(old, new)
    with open("main.py", "w") as f:
        f.write(content)
    print("Updated main.py successfully.")
else:
    print("Could not find the exact line — main.py may differ from expected. Check manually.")
