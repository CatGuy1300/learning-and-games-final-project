import json

path = "src/engine/sweeper.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Add imports for rich.progress
if "from rich.progress" not in content:
    content = content.replace("from rich.console import Console", "from rich.console import Console\nfrom rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn")

# Find the loop that dispatches workers and sets conf.execution.quiet
content = content.replace(
    "res = pool.apply_async(run_single_config, (conf.model_dump(), overrides, sweep_session_id))",
    "conf.execution.quiet = True\n            res = pool.apply_async(run_single_config, (conf.model_dump(), overrides, sweep_session_id))"
)

# Find the loop that gathers results and wrap it in a progress bar
target_loop = """        for c_hash, res in results:
            try:
                _, metrics = res.get()
                state[c_hash]["status"] = "DONE"
                state[c_hash]["metrics"] = metrics
            except Exception as e:
                console.print(f"[bold red]Config {c_hash} FAILED:[/bold red] {e}")
                state[c_hash]["status"] = "FAILED"
                state[c_hash]["error"] = str(e)
                
            # Update state file dynamically
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)"""

replacement_loop = """        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Sweeping Hyperparameters[/bold cyan]"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "({task.completed}/{task.total})",
            "•",
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("Running...", total=len(pending_configs))
            
            for c_hash, res in results:
                try:
                    _, metrics = res.get()
                    state[c_hash]["status"] = "DONE"
                    state[c_hash]["metrics"] = metrics
                except Exception as e:
                    console.print(f"[bold red]Config {c_hash} FAILED:[/bold red] {e}")
                    state[c_hash]["status"] = "FAILED"
                    state[c_hash]["error"] = str(e)
                    
                # Update state file dynamically
                with open(state_file, "w") as f:
                    json.dump(state, f, indent=2)
                progress.advance(task)"""

content = content.replace(target_loop, replacement_loop)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
