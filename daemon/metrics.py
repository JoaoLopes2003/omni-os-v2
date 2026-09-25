import time
import json
import os

class MetricsTracker:
    def __init__(self, output_dir: str, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.output_dir = output_dir
        self.model_calls = []
        self.action_executions = []
        self.system_operations = []
        
        self.PRICING = {
            "gemini-3.8-flash": {
                "input": 0.75,
                "output": 3.75
            },
            "gemini-3.1-pro-preview": {
                "input": 2.00,
                "output": 12.00
            }
        }

        self.start_time = time.time()
        self.first_action_time = None
        
        self.cache_hits = 0
        self.cache_misses = 0

    def log_system_op(self, operation_name: str, duration: float):
        """Logs local compute tasks like OCR, Image Resizing, or DB lookups."""
        if not self.debug_mode: return
        self.system_operations.append({
            "operation": operation_name,
            "duration": round(duration, 3)
        })

    def log_model_call(self, agent: str, model_name: str, duration: float, input_tokens: int, output_tokens: int):
        """Logs an API call. `agent` should be 'Master', 'Planner', or 'Mapper'."""
        if not self.debug_mode: return
        rates = self.PRICING.get(model_name, {"input": 0.0, "output": 0.0}) # Fallback if model not found
        
        cost = ((input_tokens / 1_000_000) * rates["input"]) + \
               ((output_tokens / 1_000_000) * rates["output"])
        
        self.model_calls.append({
            "agent": agent,
            "model": model_name,
            "duration": round(duration, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        })

    def log_action(self, action_type: str, duration: float):
        if not self.debug_mode: return
        if self.first_action_time is None:
            self.first_action_time = time.time() - self.start_time
            
        self.action_executions.append({
            "action_type": action_type,
            "duration": round(duration, 3)
        })

    def log_cache_event(self, hit: bool):
        if not self.debug_mode: return
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def generate_report(self):
        """Compiles the tracked data into a structured report."""
        if not self.debug_mode: return
        report_path = os.path.join(self.output_dir, "performance_metrics.md")
        
        # Calculate Global Totals
        total_time = time.time() - self.start_time
        total_cost = sum(call["cost"] for call in self.model_calls)
        total_in_tokens = sum(call["input_tokens"] for call in self.model_calls)
        total_out_tokens = sum(call["output_tokens"] for call in self.model_calls)
        
        # Aggregate Agent Stats
        agent_stats = {}
        for call in self.model_calls:
            agent = call["agent"]
            if agent not in agent_stats:
                agent_stats[agent] = {"count": 0, "time": 0.0, "in_tokens": 0, "out_tokens": 0, "cost": 0.0}
            agent_stats[agent]["count"] += 1
            agent_stats[agent]["time"] += call["duration"]
            agent_stats[agent]["in_tokens"] += call["input_tokens"]
            agent_stats[agent]["out_tokens"] += call["output_tokens"]
            agent_stats[agent]["cost"] += call["cost"]

        # Aggregate Action Stats
        action_stats = {}
        for action in self.action_executions:
            a_type = action["action_type"]
            if a_type not in action_stats:
                action_stats[a_type] = []
            action_stats[a_type].append(action["duration"])
            
        # Write Report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Omni-OS Execution Metrics\n\n")
            
            f.write("## 1. Global Performance\n")
            f.write(f"- **Total Execution Time:** {total_time:.2f}s\n")
            f.write(f"- **Time-to-First-Action (TTFA):** {self.first_action_time:.2f}s\n" if self.first_action_time else "- **TTFA:** N/A\n")
            f.write(f"- **Total API Cost:** ${total_cost:.6f}\n")
            f.write(f"- **Total Tokens:** {total_in_tokens} In / {total_out_tokens} Out\n")
            f.write(f"- **State Cache Performance:** {self.cache_hits} Hits / {self.cache_misses} Misses\n\n")
            
            f.write("## 2. Agent Averages\n")
            f.write("| Agent | Calls | Avg Time (s) | Avg In Tokens | Avg Out Tokens | Avg Cost ($) |\n")
            f.write("|---|---|---|---|---|---|\n")
            for agent, stats in agent_stats.items():
                c = stats["count"]
                f.write(f"| {agent} | {c} | {stats['time']/c:.3f} | {stats['in_tokens']/c:.0f} | {stats['out_tokens']/c:.0f} | ${stats['cost']/c:.6f} |\n")
            
            f.write("\n## 3. Chronological Model Calls\n")
            f.write("| Agent | Model | Time (s) | In Tokens | Out Tokens | Cost ($) |\n")
            f.write("|---|---|---|---|---|---|\n")
            for call in self.model_calls:
                f.write(f"| {call['agent']} | {call['model']} | {call['duration']} | {call['input_tokens']} | {call['output_tokens']} | ${call['cost']:.6f} |\n")

            f.write("\n## 4. System Operations (Local Compute)\n")
            f.write("| Operation | Count | Avg Time (s) | Total Time (s) |\n")
            f.write("|---|---|---|---|\n")
            
            sys_stats = {}
            for op in self.system_operations:
                name = op["operation"]
                if name not in sys_stats:
                    sys_stats[name] = []
                sys_stats[name].append(op["duration"])
                
            for name, durations in sys_stats.items():
                avg = sum(durations) / len(durations)
                tot = sum(durations)
                f.write(f"| {name} | {len(durations)} | {avg:.3f} | {tot:.3f} |\n")
            
            f.write("\n## 5. Physical Action Averages\n")
            f.write("| Action Type | Count | Avg Time (s) | Total Time (s) |\n")
            f.write("|---|---|---|---|\n")
            for a_type, durations in action_stats.items():
                avg = sum(durations) / len(durations)
                tot = sum(durations)
                f.write(f"| {a_type} | {len(durations)} | {avg:.3f} | {tot:.3f} |\n")