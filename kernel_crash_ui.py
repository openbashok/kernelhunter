#!/usr/bin/env python3
import os
import csv
import curses
import subprocess
import time
import sys
import importlib.util

CSV_LOG_PATH = "kernel_errors.csv"
GEN_DIR = "kernelhunter_generations"


def load_crashes():
    crashes = []
    if not os.path.exists(CSV_LOG_PATH):
        return crashes

    with open(CSV_LOG_PATH, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prog = row['Program']
            gen_prog = prog.split('_')

            if len(gen_prog) != 2:
                continue

            gen = gen_prog[0]
            prog_id = gen_prog[1]
            gen_dir = f"gen_{gen[-4:]}"

            source_path = os.path.join(GEN_DIR, gen_dir, f"{prog}.c")
            json_path = os.path.join("kernelhunter_critical", f"crash_{gen}_{prog_id}.json")

            crashes.append({
                "timestamp": row['Timestamp'],
                "program": prog,
                "type": row['Error Type'],
                "addr": row['Address'],
                "msg": row['Message'],
                "original_line": row['Original Line'],
                "source_path": source_path,
                "json_path": json_path,
                "binary_path": source_path.replace('.c', '')
            })

    return crashes


def draw_table(stdscr, crashes, selected_idx, sort_key, reverse, offset, search_term):
    stdscr.clear()
    height, width = stdscr.getmaxyx()

    title = f"KernelHunter - Crash Explorer [Sort: {sort_key} {'DESC' if reverse else 'ASC'} | Search: {search_term or 'N/A'}]"
    stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)

    headers = ["Timestamp", "Program", "Error type", "Address", "Message"]
    header_str = " | ".join(h.ljust(15) for h in headers)
    stdscr.addstr(2, 2, header_str, curses.A_UNDERLINE)

    visible_crashes = crashes[offset:offset + height - 6]
    for i, crash in enumerate(visible_crashes):
        line = f"{crash['timestamp']:<15} | {crash['program']:<15} | {crash['type']:<15} | {crash['addr']:<15} | {crash['msg'][:30]}"
        if i + offset == selected_idx:
            stdscr.addstr(i + 3, 2, line, curses.A_REVERSE)
        else:
            stdscr.addstr(i + 3, 2, line)

    stdscr.addstr(height - 2, 2, "↑↓ navigate | Enter: options | q: quit | p: program | t: type | a: addr | m: message | /: search")
    stdscr.refresh()


def show_options_menu(crash):
    while True:
        os.system('clear')
        print(f"\nCrash: {crash['program']}")
        print("=" * 40)
        print("[1] View source code")
        print("[2] View stderr (JSON if available)")
        print("[3] Run GDB on binary")
        print("[4] Run binary 100 times")
        print("[5] Analyze with Valgrind (memory leaks)")
        print("[6] Disassemble binary (objdump)")
        print("[7] List symbols (nm)")
        print("[8] Trace system calls (strace)")
        print("[9] Trace library calls (ltrace)")
        print("[10] Analyze ELF structure (readelf)")
        print("[11] Generate complete diagnostic report")
        print("[12] Analyze with GPT")
        print("[q] Back\n")

        opt = input("Select an option: ").strip().lower()
        if opt == '1':
            os.system(f"less {crash['source_path']}")
        elif opt == '2':
            if os.path.exists(crash['json_path']):
                os.system(f"less {crash['json_path']}")
            else:
                print("JSON file not found.")
                input("Press ENTER to continue...")
        elif opt == '3':
            if os.path.exists(crash['binary_path']):
                subprocess.run(["gdb", crash['binary_path']])
            else:
                print(f"Binary not found: {crash['binary_path']}")
                input("Press ENTER to continue...")
        elif opt == '4':
            if os.path.exists(crash['binary_path']):
                print(f"Running {crash['binary_path']} 100 times...")
                crashes = 0
                for i in range(100):
                    print(f"Execution {i+1}/100", end="\r")
                    try:
                        result = subprocess.run([crash['binary_path']], 
                                               stdout=subprocess.DEVNULL, 
                                               stderr=subprocess.PIPE, 
                                               timeout=1)
                        if result.returncode != 0:
                            crashes += 1
                    except subprocess.TimeoutExpired:
                        crashes += 1
                    except Exception as e:
                        print(f"Error: {e}")
                print(f"\nCompleted. The binary failed in {crashes} out of 100 executions.")
                input("Press ENTER to continue...")
            else:
                print(f"Binary not found: {crash['binary_path']}")
                input("Press ENTER to continue...")
        elif opt == '5':
            if os.path.exists(crash['binary_path']):
                print("Running Valgrind for memory error detection...")
                subprocess.run(["valgrind", "--leak-check=full", "--show-leak-kinds=all", 
                               "--track-origins=yes", crash['binary_path']])
                input("Press ENTER to continue...")
            else:
                print(f"Binary not found: {crash['binary_path']}")
                input("Press ENTER to continue...")
        elif opt == '6':
            if os.path.exists(crash['binary_path']):
                print("Disassembling binary with objdump...")
                subprocess.run(["objdump", "-d", "-S", crash['binary_path']])
                input("Press ENTER to continue...")
            else:
                print(f"Binary not found: {crash['binary_path']}")
                input("Press ENTER to continue...")
        elif opt == '7':
            if os.path.exists(crash['binary_path']):
                print("Listing symbols with nm...")
                subprocess.run(["nm", "-C", crash['binary_path']])
                input("Press ENTER to continue...")
            else:
                print(f"Binary not found: {crash['binary_path']}")
                input("Press ENTER to continue...")
        elif opt == '8':
            if os.path.exists(crash['binary_path']):
                print("Tracing system calls with strace...")
                subprocess.run(["strace", "-f", crash['binary_path']])
                input("Press ENTER to continue...")
            else:
                print(f"Binary not found: {crash['binary_path']}")
                input("Press ENTER to continue...")
        elif opt == '9':
            if os.path.exists(crash['binary_path']):
                print("Tracing library calls with ltrace...")
                subprocess.run(["ltrace", crash['binary_path']])
                input("Press ENTER to continue...")
            else:
                print(f"Binary not found: {crash['binary_path']}")
                input("Press ENTER to continue...")
        elif opt == '10':
            if os.path.exists(crash['binary_path']):
                print("Analyzing ELF structure with readelf...")
                subprocess.run(["readelf", "-a", crash['binary_path']])
                input("Press ENTER to continue...")
            else:
                print(f"Binary not found: {crash['binary_path']}")
                input("Press ENTER to continue...")
        elif opt == '11':
            if os.path.exists(crash['binary_path']):
                report_path = generate_diagnostic_report(crash)
                input("Press ENTER to continue...")
            else:
                print(f"Binary not found: {crash['binary_path']}")
                input("Press ENTER to continue...")
        elif opt == '12':
            if os.path.exists(crash['binary_path']):
                # Check if the AI module is available
                ia_module_path = "/opt/kernelhunter/kernelhunter_ia.py"
                if not os.path.exists(ia_module_path):
                    print(f"AI module not found: {ia_module_path}")
                    print("Please make sure kernelhunter_ia.py is in the same directory.")
                    input("Press ENTER to continue...")
                else:
                    # First generate a report if one doesn't exist
                    print("First generating a diagnostic report...")
                    report_path = generate_diagnostic_report(crash)
                    
                    # Check if OpenAI API key is set
                    if not os.getenv("OPENAI_API_KEY"):
                        api_key = input("Please enter your OpenAI API key: ").strip()
                        os.environ["OPENAI_API_KEY"] = api_key
                    
                    print("\nStarting AI analysis with GPT...")
                    try:
                        # Import the AI module
                        spec = importlib.util.spec_from_file_location("kernelhunter_ia", ia_module_path)
                        ia_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(ia_module)
                        
                        # Call the analyze function
                        analysis_path = ia_module.analyze_with_gpt(report_path)
                        
                        if analysis_path:
                            print(f"AI analysis complete. Results saved to {analysis_path}")
                        else:
                            print("AI analysis failed.")
                        
                        input("Press ENTER to continue...")
                    except Exception as e:
                        print(f"Error during AI analysis: {e}")
                        input("Press ENTER to continue...")
            else:
                print(f"Binary not found: {crash['binary_path']}")
                input("Press ENTER to continue...")
        elif opt == 'q':
            break

def generate_diagnostic_report_silent(crash):
    """Generate a diagnostic report without displaying or opening it"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    report_dir = "diagnostic_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, f"report_{crash['program']}_{timestamp}.txt")
    
    with open(report_file, 'w') as f:
        # Introduction
        f.write("=" * 80 + "\n")
        f.write("KERNELHUNTER AUTOMATED DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("This report contains automated diagnostics based on various analysis tools.\n")
        f.write("Its purpose is to assist in identifying kernel-level issues in the operating system.\n")
        f.write("The information provided is technical in nature and intended for diagnostic purposes only.\n\n")
        
        # Binary information
        f.write("-" * 80 + "\n")
        f.write("BINARY INFORMATION\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"Program: {crash['program']}\n")
        f.write(f"Error Type: {crash['type']}\n")
        f.write(f"Address: {crash['addr']}\n")
        f.write(f"Error Message: {crash['msg']}\n")
        f.write(f"Source Path: {crash['source_path']}\n")
        f.write(f"Binary Path: {crash['binary_path']}\n")
        f.write(f"Timestamp: {crash['timestamp']}\n\n")
        
        # Source code
        if os.path.exists(crash['source_path']):
            f.write("-" * 80 + "\n")
            f.write("SOURCE CODE\n")
            f.write("-" * 80 + "\n\n")
            try:
                with open(crash['source_path'], 'r') as source_file:
                    f.write(source_file.read())
                f.write("\n\n")
            except Exception as e:
                f.write(f"Error reading source code: {e}\n\n")
        
        # JSON error data
        if os.path.exists(crash['json_path']):
            f.write("-" * 80 + "\n")
            f.write("ERROR DATA (JSON)\n")
            f.write("-" * 80 + "\n\n")
            try:
                with open(crash['json_path'], 'r') as json_file:
                    f.write(json_file.read())
                f.write("\n\n")
            except Exception as e:
                f.write(f"Error reading JSON file: {e}\n\n")
        
        # Run binary 100 times
        f.write("-" * 80 + "\n")
        f.write("BINARY STABILITY TEST (100 EXECUTIONS)\n")
        f.write("-" * 80 + "\n\n")
        crashes = 0
        for i in range(100):
            try:
                result = subprocess.run(
                    [crash['binary_path']], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.PIPE, 
                    timeout=1
                )
                if result.returncode != 0:
                    crashes += 1
            except subprocess.TimeoutExpired:
                crashes += 1
            except Exception as e:
                f.write(f"Error during execution: {e}\n")
        
        f.write(f"Failed executions: {crashes}/100\n")
        f.write(f"Success rate: {100-crashes}%\n\n")
        
        # Valgrind analysis
        f.write("-" * 80 + "\n")
        f.write("VALGRIND MEMORY ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        try:
            result = subprocess.run(
                ["valgrind", "--leak-check=full", "--show-leak-kinds=all", "--track-origins=yes", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )
            f.write(result.stderr)
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running Valgrind: {e}\n\n")
        
        # objdump analysis
        f.write("-" * 80 + "\n")
        f.write("OBJDUMP DISASSEMBLY\n")
        f.write("-" * 80 + "\n\n")
        try:
            result = subprocess.run(
                ["objdump", "-d", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            f.write(result.stdout)
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running objdump: {e}\n\n")
        
        # nm symbol analysis
        f.write("-" * 80 + "\n")
        f.write("NM SYMBOL ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        try:
            result = subprocess.run(
                ["nm", "-C", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            f.write(result.stdout)
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running nm: {e}\n\n")
        
        # readelf analysis
        f.write("-" * 80 + "\n")
        f.write("READELF ELF STRUCTURE ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        try:
            result = subprocess.run(
                ["readelf", "-a", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            f.write(result.stdout)
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running readelf: {e}\n\n")
        
        # strace analysis (with last lines)
        f.write("-" * 80 + "\n")
        f.write("STRACE SYSTEM CALL ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        try:
            # First, get the statistics summary
            result = subprocess.run(
                ["strace", "-c", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            f.write("SUMMARY STATISTICS:\n")
            f.write(result.stderr)
            f.write("\n\n")
            
            # Then, capture the last 100 lines of actual calls
            result = subprocess.run(
                ["strace", "-f", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            
            # Get the last 100 lines (or all if less than 100)
            strace_lines = result.stderr.splitlines()
            last_lines = strace_lines[-100:] if len(strace_lines) > 100 else strace_lines
            
            f.write("LAST SYSTEM CALLS BEFORE CRASH:\n")
            for line in last_lines:
                f.write(line + "\n")
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running strace: {e}\n\n")
        
        # ltrace analysis (with last lines)
        f.write("-" * 80 + "\n")
        f.write("LTRACE LIBRARY CALL ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        try:
            # First, get the statistics summary
            result = subprocess.run(
                ["ltrace", "-c", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            f.write("SUMMARY STATISTICS:\n")
            f.write(result.stderr + result.stdout)
            f.write("\n\n")
            
            # Then, capture the last 100 lines of actual calls
            result = subprocess.run(
                ["ltrace", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            
            # Get the last 100 lines (or all if less than 100)
            ltrace_output = result.stderr + result.stdout
            ltrace_lines = ltrace_output.splitlines()
            last_lines = ltrace_lines[-100:] if len(ltrace_lines) > 100 else ltrace_lines
            
            f.write("LAST LIBRARY CALLS BEFORE CRASH:\n")
            for line in last_lines:
                f.write(line + "\n")
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running ltrace: {e}\n\n")
        
        # Summary
        f.write("=" * 80 + "\n")
        f.write("REPORT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Binary: {crash['program']}\n")
        f.write(f"Error Type: {crash['type']}\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Stability test: {100-crashes}% successful ({crashes} failures out of 100 runs)\n")
        f.write("\nThis report contains automated diagnostics and may require expert interpretation.\n")
        f.write("For detailed analysis, review individual sections above.\n")
    
    return report_file
    
def generate_diagnostic_report(crash):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    report_dir = "diagnostic_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, f"report_{crash['program']}_{timestamp}.txt")
    
    print(f"Generating comprehensive diagnostic report for {crash['program']}...")
    print(f"This may take some time. Please wait...")
    
    with open(report_file, 'w') as f:
        # Introduction
        f.write("=" * 80 + "\n")
        f.write("KERNELHUNTER AUTOMATED DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("This report contains automated diagnostics based on various analysis tools.\n")
        f.write("Its purpose is to assist in identifying kernel-level issues in the operating system.\n")
        f.write("The information provided is technical in nature and intended for diagnostic purposes only.\n\n")
        
        # Binary information
        f.write("-" * 80 + "\n")
        f.write("BINARY INFORMATION\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"Program: {crash['program']}\n")
        f.write(f"Error Type: {crash['type']}\n")
        f.write(f"Address: {crash['addr']}\n")
        f.write(f"Error Message: {crash['msg']}\n")
        f.write(f"Source Path: {crash['source_path']}\n")
        f.write(f"Binary Path: {crash['binary_path']}\n")
        f.write(f"Timestamp: {crash['timestamp']}\n\n")
        
        # Source code
        if os.path.exists(crash['source_path']):
            f.write("-" * 80 + "\n")
            f.write("SOURCE CODE\n")
            f.write("-" * 80 + "\n\n")
            try:
                with open(crash['source_path'], 'r') as source_file:
                    f.write(source_file.read())
                f.write("\n\n")
            except Exception as e:
                f.write(f"Error reading source code: {e}\n\n")
        
        # JSON error data
        if os.path.exists(crash['json_path']):
            f.write("-" * 80 + "\n")
            f.write("ERROR DATA (JSON)\n")
            f.write("-" * 80 + "\n\n")
            try:
                with open(crash['json_path'], 'r') as json_file:
                    f.write(json_file.read())
                f.write("\n\n")
            except Exception as e:
                f.write(f"Error reading JSON file: {e}\n\n")
        
        # Run binary 100 times
        f.write("-" * 80 + "\n")
        f.write("BINARY STABILITY TEST (100 EXECUTIONS)\n")
        f.write("-" * 80 + "\n\n")
        crashes = 0
        for i in range(100):
            try:
                result = subprocess.run(
                    [crash['binary_path']], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.PIPE, 
                    timeout=1
                )
                if result.returncode != 0:
                    crashes += 1
            except subprocess.TimeoutExpired:
                crashes += 1
            except Exception as e:
                f.write(f"Error during execution: {e}\n")
        
        f.write(f"Failed executions: {crashes}/100\n")
        f.write(f"Success rate: {100-crashes}%\n\n")
        
        # Valgrind analysis
        f.write("-" * 80 + "\n")
        f.write("VALGRIND MEMORY ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        try:
            result = subprocess.run(
                ["valgrind", "--leak-check=full", "--show-leak-kinds=all", "--track-origins=yes", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )
            f.write(result.stderr)
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running Valgrind: {e}\n\n")
        
        # objdump analysis
        f.write("-" * 80 + "\n")
        f.write("OBJDUMP DISASSEMBLY\n")
        f.write("-" * 80 + "\n\n")
        try:
            result = subprocess.run(
                ["objdump", "-d", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            f.write(result.stdout)
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running objdump: {e}\n\n")
        
        # nm symbol analysis
        f.write("-" * 80 + "\n")
        f.write("NM SYMBOL ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        try:
            result = subprocess.run(
                ["nm", "-C", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            f.write(result.stdout)
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running nm: {e}\n\n")
        
        # readelf analysis
        f.write("-" * 80 + "\n")
        f.write("READELF ELF STRUCTURE ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        try:
            result = subprocess.run(
                ["readelf", "-a", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            f.write(result.stdout)
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running readelf: {e}\n\n")
        
        # strace analysis (with last lines)
        f.write("-" * 80 + "\n")
        f.write("STRACE SYSTEM CALL ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        try:
            # First, get the statistics summary
            result = subprocess.run(
                ["strace", "-c", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            f.write("SUMMARY STATISTICS:\n")
            f.write(result.stderr)
            f.write("\n\n")
            
            # Then, capture the last 100 lines of actual calls
            result = subprocess.run(
                ["strace", "-f", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            
            # Get the last 100 lines (or all if less than 100)
            strace_lines = result.stderr.splitlines()
            last_lines = strace_lines[-100:] if len(strace_lines) > 100 else strace_lines
            
            f.write("LAST SYSTEM CALLS BEFORE CRASH:\n")
            for line in last_lines:
                f.write(line + "\n")
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running strace: {e}\n\n")
        
        # ltrace analysis (with last lines)
        f.write("-" * 80 + "\n")
        f.write("LTRACE LIBRARY CALL ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        try:
            # First, get the statistics summary
            result = subprocess.run(
                ["ltrace", "-c", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            f.write("SUMMARY STATISTICS:\n")
            f.write(result.stderr + result.stdout)
            f.write("\n\n")
            
            # Then, capture the last 100 lines of actual calls
            result = subprocess.run(
                ["ltrace", crash['binary_path']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            
            # Get the last 100 lines (or all if less than 100)
            ltrace_output = result.stderr + result.stdout
            ltrace_lines = ltrace_output.splitlines()
            last_lines = ltrace_lines[-100:] if len(ltrace_lines) > 100 else ltrace_lines
            
            f.write("LAST LIBRARY CALLS BEFORE CRASH:\n")
            for line in last_lines:
                f.write(line + "\n")
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running ltrace: {e}\n\n")
        
        # Summary
        f.write("=" * 80 + "\n")
        f.write("REPORT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Binary: {crash['program']}\n")
        f.write(f"Error Type: {crash['type']}\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Stability test: {100-crashes}% successful ({crashes} failures out of 100 runs)\n")
        f.write("\nThis report contains automated diagnostics and may require expert interpretation.\n")
        f.write("For detailed analysis, review individual sections above.\n")
    
    print(f"\nDiagnostic report generated successfully!")
    print(f"Report saved to: {report_file}")
    
    # Open the report file automatically
    try:
        if os.name == 'nt':  # Windows
            os.startfile(report_file)
        elif os.name == 'posix':  # Linux/Unix/MacOS
            if os.system('which xdg-open > /dev/null') == 0:
                subprocess.call(['xdg-open', report_file])
            elif os.system('which open > /dev/null') == 0:  # MacOS
                subprocess.call(['open', report_file])
            else:
                subprocess.call(['less', report_file])  # Fallback to less
        print(f"\nThe report has been opened for viewing.")
        print(f"It is also saved at: {os.path.abspath(report_file)}")
    except Exception as e:
        print(f"\nCouldn't automatically open the report: {e}")
        print(f"You can find it at: {os.path.abspath(report_file)}")
    
    return report_file


def filter_crashes(crashes, term):
    if not term:
        return crashes
    term = term.lower()
    return [c for c in crashes if 
            term in c['program'].lower() or 
            term in c['type'].lower() or 
            term in c['addr'].lower() or 
            term in c['msg'].lower()]


def main(stdscr):
    curses.curs_set(0)
    sort_key = "timestamp"
    reverse = False
    selected_idx = 0
    offset = 0
    search_term = ""
    last_load = 0
    crashes = load_crashes()

    while True:
        if time.time() - last_load > 2:
            crashes = load_crashes()
            last_load = time.time()

        filtered_crashes = filter_crashes(crashes, search_term)
        filtered_crashes.sort(key=lambda c: c[sort_key], reverse=reverse)
        selected_idx = max(0, min(selected_idx, len(filtered_crashes) - 1))
        offset = max(0, min(offset, max(0, len(filtered_crashes) - 1)))

        draw_table(stdscr, filtered_crashes, selected_idx, sort_key, reverse, offset, search_term)

        key = stdscr.getch()
        if key == curses.KEY_UP and selected_idx > 0:
            selected_idx -= 1
            if selected_idx < offset:
                offset -= 1
        elif key == curses.KEY_DOWN and selected_idx < len(filtered_crashes) - 1:
            selected_idx += 1
            height, _ = stdscr.getmaxyx()
            if selected_idx >= offset + height - 6:
                offset += 1
        elif key == ord('q'):
            break
        elif key == ord('\n'):
            curses.endwin()
            show_options_menu(filtered_crashes[selected_idx])
        elif key in [ord('p'), ord('t'), ord('a'), ord('m')]:
            keys = {'p': 'program', 't': 'type', 'a': 'addr', 'm': 'msg'}
            pressed_key = chr(key)
            if sort_key == keys[pressed_key]:
                reverse = not reverse
            else:
                sort_key = keys[pressed_key]
                reverse = False
        elif key == ord('/'):
            curses.echo()
            stdscr.addstr(curses.LINES - 1, 0, "Search: ")
            search_term = stdscr.getstr().decode('utf-8')
            curses.noecho()


if __name__ == "__main__":
    curses.wrapper(main)