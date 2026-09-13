path = "src/engine/sweeper.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target = """    base_config_path = sweep_data["base_config"]
    grid = sweep_data.get("grid", {})
    
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f) or {}"""

replacement = """    mode = sweep_data.get("mode", "grid")
    
    base_config_path = sweep_data["base_config"]
    grid = sweep_data.get("grid", {})
    
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f) or {}
        
    if mode == "optuna":
        sweep_data["base_config"] = base_raw
        return run_optuna_sweep(sweep_data, workers)"""

content = content.replace(target, replacement)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
