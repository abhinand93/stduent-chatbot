import yaml

def list_to_bullets(items):
    return "\n- ".join(items)


def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

