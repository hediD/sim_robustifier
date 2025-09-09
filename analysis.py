# analysis.py
from typing import Dict, List, Tuple

def summarize_core_models(results: Dict, baseline_key: str = "original"):
    print("\nCORE MODEL COMPARISON:")
    print(f"{'Model':<20} {'Real Tank Acc':<15} {'Sim Tank Acc':<15} {'Real Improvement':<18}")
    print("-" * 70)

    base_real = results[baseline_key]["real"]["tank_accuracy"]
    for name in [baseline_key, "imagenet_only", "sim_only", "imagenet_sim"]:
        if name in results:
            real_acc = results[name]["real"]["tank_accuracy"]
            sim_acc  = results[name]["sim"]["tank_accuracy"]
            delta = real_acc - base_real
            print(f"{name:<20} {real_acc:<15.4f} {sim_acc:<15.4f} {delta:+.4f}")

def summarize_env_models(results: Dict, env_names: List[str], baseline_key: str = "original") -> List[Tuple[str, float, float, str]]:
    print(f"\nENVIRONMENT COMPARISON:")
    print(f"{'Environment':<15} {'Real Tank Acc':<15} {'vs Baseline':<12} {'Notes':<20}")
    print("-" * 65)
    base_real = results[baseline_key]["real"]["tank_accuracy"]

    env_rows: List[Tuple[str, float, float, str]] = []
    for env in env_names:
        if env in results:
            real_acc = results[env]["real"]["tank_accuracy"]
            improvement = real_acc - base_real
            if env == "sim_black":
                notes = "(eval env)"
            elif env == "all_combined":
                notes = "(training mix)"
            else:
                notes = "(individual)"
            env_rows.append((env, real_acc, improvement, notes))

    env_rows.sort(key=lambda x: x[1], reverse=True)
    for i, (env, acc, imp, notes) in enumerate(env_rows):
        star = " ⭐" if i == 0 else ""
        print(f"{env:<15} {acc:<15.4f} {imp:+.4f}    {notes:<20}{star}")
    return env_rows

def key_insights(results: Dict):
    print("\nKEY INSIGHTS:")
    if "sim_only" in results and "imagenet_only" in results:
        sim_real = results["sim_only"]["real"]["tank_accuracy"]
        img_real = results["imagenet_only"]["real"]["tank_accuracy"]
        diff = sim_real - img_real
        print(f"  • Sim-only vs ImageNet-only: {diff:+.4f} ({diff*100:+.2f}%)")
        if diff > 0.02:
            print("    ✅ Simulated data significantly outperforms ImageNet!")
        elif diff > -0.02:
            print("    ➡️ Simulated and ImageNet data perform similarly")
        else:
            print("    ❌ ImageNet data outperforms simulated data")

    if "imagenet_sim" in results and "sim_only" in results:
        combo = results["imagenet_sim"]["real"]["tank_accuracy"]
        sim_o = results["sim_only"]["real"]["tank_accuracy"]
        gain = combo - sim_o
        print(f"  • Adding ImageNet to sim data: {gain:+.4f} ({gain*100:+.2f}%)")
        if gain > 0.01:
            print("    ✅ ImageNet + Sim combination is beneficial!")
        elif gain > -0.01:
            print("    ➡️ Combination provides minimal benefit")
        else:
            print("    ❌ Adding ImageNet hurts performance")
