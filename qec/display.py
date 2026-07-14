"""Console circuit printing (handles Windows cp1252 stdout choking on box-drawing chars)."""
import sys


def print_circuit(qc, title=None, fold=120):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if title:
        print(f"\n--- {title} ---")
    print(qc.draw(output="text", fold=fold))
    print(f"(depth={qc.depth()}, gate count={qc.size()}, qubits={qc.num_qubits})")
