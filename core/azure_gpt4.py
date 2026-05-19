import json
import re
import time
import os
import openai
import configparser
import base64
from colorama import Fore
from typing import List


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _load_config():
    config = configparser.ConfigParser()
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "config.ini"),
        "../config/config.ini",
        "config/config.ini",
    ]
    for path in candidates:
        if os.path.exists(path):
            config.read(path, encoding="utf-8")
            return config
    raise FileNotFoundError("config.ini not found in any of: " + ", ".join(candidates))


def ask_gpt4o(system_prompt, user_prompt, images: List[str], need_json=True) -> dict:
    """Backed by DeepSeek (OpenAI-compatible chat API).

    DeepSeek has no vision endpoint, so any image paths passed in are
    ignored with a warning. The function name is preserved so callers
    in agent_semantic / agent_execute keep working without edits.
    """
    config = _load_config()

    ak = config.get('gpt4', 'gpt_key')
    endpoint = config.get('gpt4', 'endpoint')
    model_name = config.get('gpt4', 'model', fallback='deepseek-chat')
    max_tokens = 4096

    if images:
        print(Fore.YELLOW + f"[ask_gpt4o] DeepSeek has no vision API; ignoring {len(images)} image(s)." + Fore.RESET)

    client = openai.OpenAI(api_key=ak, base_url=endpoint)

    max_attempts = 5
    attempt = 0
    while attempt < max_attempts:
        try:
            kwargs = dict(
                model=model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )
            if need_json:
                kwargs["response_format"] = {"type": "json_object"}
            completion = client.chat.completions.create(**kwargs)
        except Exception as e:
            print(Fore.RED + "deepseek Exception: " + str(e) + Fore.RESET)
            if 'rate' in str(e).lower() or 'qpm' in str(e).lower():
                time.sleep(20)
            else:
                time.sleep(5)
            attempt += 1
            if attempt == max_attempts:
                return str(e) + " - deepseek call failed"
            continue

        response = json.loads(completion.model_dump_json())
        text = response['choices'][0]['message']['content']
        if not need_json:
            return text
        try:
            return json.loads(text)
        except Exception:
            pattern = r'\{[^}]*\}'
            matches = re.findall(pattern, text)
            if matches:
                try:
                    return json.loads(matches[0])
                except Exception as e:
                    print(Fore.RED + "json parse Exception: " + str(e) + Fore.RESET)
            attempt += 1
    return "None"
