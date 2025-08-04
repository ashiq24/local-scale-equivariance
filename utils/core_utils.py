def get_wandb_api_key(api_key_file="config/wandb_api_key.txt"):
    with open(api_key_file, "r") as f:
        key = f.read()
    return key.strip()
