import json
import sys

def json_to_zon(data, indent=0):
    space = " " * (indent * 2)
    if data is None:
        return "null"
    elif isinstance(data, bool):
        return str(data).lower()
    elif isinstance(data, (int, float)):
        return str(data)
    elif isinstance(data, str):
        escaped = data.replace("\\", "\\\\").replace(""", "\\\"")
        return """ + escaped + """
    elif isinstance(data, list):
        res = ".{\n"
        for item in data:
            res += space + "  " + json_to_zon(item, indent + 1) + ",\n"
        res += space + "}"
        return res
    elif isinstance(data, dict):
        res = ".{\n"
        for k, v in data.items():
            k_escaped = k.replace("\\", "\\\\").replace(""", "\\\"")
            res += space + "  .@\"" + k_escaped + "\" = " + json_to_zon(v, indent + 1) + ",\n"
        res += space + "}"
        return res
    return str(data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    with open(sys.argv[1], "r") as f:
        data = json.load(f)
    print(json_to_zon(data))