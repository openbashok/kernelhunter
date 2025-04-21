#!/usr/bin/env python3
import os
import requests
import json
import time
import argparse
import sys

# API key and endpoint information
# You need to set your OpenAI API key as an environment variable
# export OPENAI_API_KEY=your_api_key_here
API_KEY = os.getenv("OPENAI_API_KEY")
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

def analyze_report(report_path):
    """
    Analyze a diagnostic report using the OpenAI API
    """
    if not API_KEY:
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        print("Example: export OPENAI_API_KEY=your_api_key_here")
        return None
    
    # Read report file
    try:
        with open(report_path, 'r') as f:
            report_content = f.read()
    except Exception as e:
        print(f"Error reading report file: {e}")
        return None
    
    # Trim report if it's too long (token limits)
    # GPT models have token limits, so we need to focus on the most important parts
    # We'll include the first part (introduction/summary) and key analysis sections
    sections = report_content.split("-" * 80)
    
    # Always include introduction, binary info, and summary
    important_sections = [sections[0]]  # Introduction
    
    # Include these key diagnostic sections if they exist
    key_section_names = [
        "BINARY INFORMATION",
        "SOURCE CODE",  # Including a snippet of source code
        "BINARY STABILITY TEST",
        "VALGRIND MEMORY ANALYSIS",
        "LAST SYSTEM CALLS BEFORE CRASH",
        "LAST LIBRARY CALLS BEFORE CRASH"
    ]
    
    for section in sections:
        for key_name in key_section_names:
            if key_name in section:
                # Take first 1000 characters for large sections to limit tokens
                if len(section) > 10000 and (key_name == "SOURCE CODE" or 
                                           key_name == "VALGRIND MEMORY ANALYSIS"):
                    important_sections.append(section[:10000] + "\n...[truncated]...")
                else:
                    important_sections.append(section)
                break
    
    # Prepare the prompt for OpenAI
    # We'll direct the analysis with specific instructions
    system_prompt = """
    Eres un experto en análisis de vulnerabilidades del kernel y fuzzing de sistemas operativos.

Tu tarea es analizar reportes automáticos generados por la herramienta KernelHunter, que ejecuta shellcodes en espacio de usuario para intentar identificar vulnerabilidades o fallos graves que afecten al kernel del sistema operativo.

Debes seguir estos pasos para cada análisis:

1. Lee atentamente la información proporcionada sobre el crash report.
2. Identifica claramente si el problema ocurrió exclusivamente en espacio de usuario o si involucra llamadas al kernel que puedan afectar su estabilidad.
3. Revisa la evidencia crítica, en particular:
   - Resultados de Valgrind (errores de memoria graves).
   - Últimas syscalls ejecutadas antes del crash (strace).
   - Código ejecutado (shellcode).
   - Frecuencia y estabilidad del crash.

4. Clasifica explícitamente el problema según su gravedad en una de estas categorías:
   - GRAVEDAD: ALTA (potencial riesgo de corrupción del kernel, escalada de privilegios, instrucciones privilegiadas ejecutadas).
   - GRAVEDAD: MEDIA (llamadas sospechosas al kernel, instrucciones anómalas pero sin evidencia directa de daño al kernel).
   - GRAVEDAD: BAJA (crash en espacio usuario, bien gestionado por el kernel, no representa riesgo real).

5. Da una explicación técnica breve pero clara justificando tu clasificación.

Usa texto plano (ASCII) sin markdown en tu respuesta.
    """
    
    # Join the important sections back together
    filtered_report = "\n" + "-"*80 + "\n".join(important_sections)
    
    # Prepare the message payload for the API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is a KernelHunter crash report to analyze:\n\n{filtered_report}"}
    ]
    
    # Make the API request
    print("Sending report to AI for analysis...")
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        payload = {
            "model": "gpt-4o",  # Using GPT-4 for advanced technical analysis
            "messages": messages,
            "temperature": 0.3,  # Lower temperature for more focused analysis
            "max_tokens": 10000   # Adjust as needed
        }
        
        response = requests.post(API_ENDPOINT, headers=headers, json=payload)
        response_data = response.json()
        
        if response.status_code == 200 and "choices" in response_data:
            analysis = response_data["choices"][0]["message"]["content"]
            return analysis
        else:
            print(f"API Error: {response.status_code}")
            print(response_data)
            return None
    
    except Exception as e:
        print(f"Error communicating with OpenAI API: {e}")
        return None

def save_analysis(analysis, original_report_path):
    """
    Save the AI analysis to a file
    """
    if not analysis:
        return None
    
    # Create a filename based on the original report
    report_dir = os.path.dirname(original_report_path)
    report_name = os.path.basename(original_report_path)
    analysis_path = os.path.join(report_dir, f"ai_analysis_{report_name}")
    
    try:
        with open(analysis_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("KERNELHUNTER AI CRASH ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Original report: {original_report_path}\n")
            f.write(f"Analysis date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("-" * 80 + "\n")
            f.write("AI DIAGNOSTIC ANALYSIS\n")
            f.write("-" * 80 + "\n\n")
            f.write(analysis)
            f.write("\n\n")
            f.write("-" * 80 + "\n")
            f.write("NOTE: This analysis was generated by an AI assistant and should be used\n")
            f.write("as a diagnostic aid only. Human expertise is recommended for final evaluation.\n")
    
        print(f"AI analysis saved to: {analysis_path}")
        return analysis_path
    
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return None

def analyze_with_gpt(report_path):
    """
    Main function to analyze a report with GPT and save the results
    """
    # Validate report path
    if not os.path.exists(report_path):
        print(f"Error: Report file not found at {report_path}")
        return None
    
    # Get AI analysis
    analysis = analyze_report(report_path)
    
    if not analysis:
        print("Failed to generate AI analysis.")
        return None
    
    # Save the analysis
    analysis_path = save_analysis(analysis, report_path)
    return analysis_path

# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze KernelHunter crash reports using AI")
    parser.add_argument("report_path", help="Path to the KernelHunter diagnostic report")
    args = parser.parse_args()
    
    analysis_path = analyze_with_gpt(args.report_path)
    
    if analysis_path:
        print(f"Analysis complete. Results saved to {analysis_path}")
        # Try to open the file
        try:
            if os.name == 'nt':  # Windows
                os.startfile(analysis_path)
            elif os.name == 'posix':  # Linux/Unix/MacOS
                if os.system('which xdg-open > /dev/null') == 0:
                    os.system(f'xdg-open "{analysis_path}"')
                elif os.system('which open > /dev/null') == 0:  # MacOS
                    os.system(f'open "{analysis_path}"')
                else:
                    os.system(f'less "{analysis_path}"')
        except Exception as e:
            print(f"Couldn't automatically open the analysis file: {e}")
    else:
        print("Analysis failed.")
        sys.exit(1)


