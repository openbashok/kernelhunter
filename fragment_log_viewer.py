# fragment_log_viewer.py

def load_fragment_log(filepath="kernelhunter_fragment_log.txt"):
    """Carga el log de fragmentos y devuelve una lista de registros."""
    if not os.path.exists(filepath):
        print("[ERROR] Log de fragmentos no encontrado.")
        return []

    with open(filepath, "r") as f:
        lines = f.readlines()

    records = []
    for line in lines:
        if "BIN:" in line and "OFFSET:" in line:
            parts = line.strip().split('|')
            record = {
                "timestamp": parts[0].split(']')[0][1:],
                "bin_path": parts[0].split('BIN:')[1].strip(),
                "offset": parts[1].split('OFFSET:')[1].strip(),
                "length": parts[2].split('LEN:')[1].strip(),
                "fragment_hex": parts[3].split('FRAGMENT_HEX:')[1].strip(),
            }
            records.append(record)

    return records

def list_fragments():
    """Lista todos los fragmentos registrados."""
    fragments = load_fragment_log()
    for idx, frag in enumerate(fragments):
        print(f"[{idx}] {frag['timestamp']} | {frag['bin_path']} | Offset: {frag['offset']} | Len: {frag['length']} | Frag: {frag['fragment_hex'][:20]}...")

def find_fragments_by_bin(binary_name):
    """Busca fragmentos extraídos de un binario específico."""
    fragments = load_fragment_log()
    for frag in fragments:
        if binary_name in frag["bin_path"]:
            print(f"{frag['timestamp']} | {frag['bin_path']} | Offset: {frag['offset']} | Len: {frag['length']} | Frag: {frag['fragment_hex'][:20]}...")

# Ejemplo de uso
if __name__ == "__main__":
    print("Listado de fragmentos disponibles:")
    list_fragments()
