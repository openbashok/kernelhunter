#!/usr/bin/env python3

import os
import subprocess

# Shellcode correcto: mov rdi, 523; ret
shellcode = b"\x48\xc7\xc7\x0b\x02\x00\x00\xc3"

def format_shellcode(shellcode):
    return ','.join(f'0x{b:02x}' for b in shellcode)

def generar_celula_c():
    code_array = format_shellcode(shellcode)

    c_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#define FUNC_HOLA_MUNDO 523

unsigned char code[] = {{{code_array}}};

void hola_mundo() {{
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
    write(1, "Hezzzzzzzzzzzzzzzzzzzzzzzzllo World!\\n", 12);
}}

void dispatch(int action) {{
    switch (action) {{
        case FUNC_HOLA_MUNDO:
            hola_mundo();
            break;
        default:
            write(1, "[-] Unknown action\\n", 20);
            break;
    }}
}}

int main() {{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    void *exec = mmap(0, sizeof(code), PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_ANON | MAP_PRIVATE, -1, 0);
    if (exec == MAP_FAILED) {{
        perror("mmap");
        return 1;
    }}

    memcpy(exec, code, sizeof(code));

    ((void(*)())exec)();  // Ejecutar el shellcode

    register int action asm("rdi");

    if (action >= 500) {{
        dispatch(action);
    }}

    munmap(exec, sizeof(code));
    return 0;
}}
"""
    with open("celula_hola.c", "w") as f:
        f.write(c_code)

    print("[+] Archivo celula_hola.c generado.")

def compilar_y_ejecutar():
    print("[*] Compilando...")
    subprocess.run(["gcc", "celula_hola.c", "-o", "celula_hola"], check=True)
    print("[+] Compilación exitosa.")
    print("[*] Ejecutando...\n")
    subprocess.run(["./celula_hola"])

if __name__ == "__main__":
    generar_celula_c()
    compilar_y_ejecutar()
