<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

// Configuración
$config = [
    'metrics_file' => '../kernelhunter_metrics.json',
    'crash_dir' => '../kernelhunter_crashes',
    'critical_dir' => '../kernelhunter_critical',
    'log_file' => '../kernelhunter_survivors.txt',
    'crash_log' => '../kernelhunter_crashes.txt',
    'generations_dir' => '../kernelhunter_generations'
];

// Router simple
$request = $_SERVER['REQUEST_URI'];
$method = $_SERVER['REQUEST_METHOD'];

// Parsear la URL
$path = parse_url($request, PHP_URL_PATH);
$path = trim($path, '/');
$segments = explode('/', $path);

// Endpoints
switch ($segments[0] ?? '') {
    case 'api':
        handleApi($segments[1] ?? '', $method);
        break;
    case 'ws':
        handleWebSocket();
        break;
    default:
        http_response_code(404);
        echo json_encode(['error' => 'Endpoint not found']);
        break;
}

function handleApi($endpoint, $method) {
    global $config;
    
    switch ($endpoint) {
        case 'metrics':
            if ($method === 'GET') {
                getMetrics($config);
            }
            break;
            
        case 'generation':
            if ($method === 'GET') {
                $genId = $_GET['id'] ?? null;
                getGeneration($config, $genId);
            }
            break;
            
        case 'crashes':
            if ($method === 'GET') {
                getCrashes($config);
            }
            break;
            
        case 'critical':
            if ($method === 'GET') {
                getCriticalCrashes($config);
            }
            break;
            
        case 'population':
            if ($method === 'GET') {
                getPopulation($config);
            }
            break;
            
        case 'shellcode':
            if ($method === 'GET') {
                $hash = $_GET['hash'] ?? null;
                getShellcodeDetails($config, $hash);
            }
            break;
            
        case 'stats':
            if ($method === 'GET') {
                getStats($config);
            }
            break;
            
        default:
            http_response_code(404);
            echo json_encode(['error' => 'API endpoint not found']);
            break;
    }
}

function getMetrics($config) {
    if (!file_exists($config['metrics_file'])) {
        echo json_encode(['error' => 'Metrics file not found']);
        return;
    }
    
    $metrics = json_decode(file_get_contents($config['metrics_file']), true);
    
    // Calcular estadísticas adicionales
    $stats = calculateStats($metrics);
    
    echo json_encode([
        'metrics' => $metrics,
        'stats' => $stats,
        'timestamp' => time()
    ]);
}

function getGeneration($config, $genId) {
    if (!$genId) {
        http_response_code(400);
        echo json_encode(['error' => 'Generation ID required']);
        return;
    }
    
    $genPath = $config['generations_dir'] . "/gen_" . str_pad($genId, 4, '0', STR_PAD_LEFT);
    
    if (!is_dir($genPath)) {
        http_response_code(404);
        echo json_encode(['error' => 'Generation not found']);
        return;
    }
    
    $files = glob($genPath . "/*.c");
    $programs = [];
    
    foreach ($files as $file) {
        $filename = basename($file);
        preg_match('/g(\d{4})_p(\d{4})\.c/', $filename, $matches);
        
        if (count($matches) === 3) {
            $programs[] = [
                'id' => $matches[2],
                'filename' => $filename,
                'content' => file_get_contents($file),
                'size' => filesize($file)
            ];
        }
    }
    
    echo json_encode([
        'generation' => $genId,
        'programs' => $programs,
        'count' => count($programs)
    ]);
}

function getCrashes($config) {
    $crashes = [];
    
    if (is_dir($config['crash_dir'])) {
        $files = glob($config['crash_dir'] . "/*.json");
        
        foreach ($files as $file) {
            $data = json_decode(file_get_contents($file), true);
            if ($data) {
                $crashes[] = $data;
            }
        }
    }
    
    // Ordenar por timestamp descendente
    usort($crashes, function($a, $b) {
        return ($b['timestamp'] ?? 0) - ($a['timestamp'] ?? 0);
    });
    
    echo json_encode([
        'crashes' => $crashes,
        'count' => count($crashes)
    ]);
}

function getCriticalCrashes($config) {
    $crashes = [];
    
    if (is_dir($config['critical_dir'])) {
        $files = glob($config['critical_dir'] . "/*.json");
        
        foreach ($files as $file) {
            $data = json_decode(file_get_contents($file), true);
            if ($data && ($data['system_impact'] ?? false)) {
                $crashes[] = $data;
            }
        }
    }
    
    // Ordenar por timestamp descendente
    usort($crashes, function($a, $b) {
        return ($b['timestamp'] ?? 0) - ($a['timestamp'] ?? 0);
    });
    
    echo json_encode([
        'critical_crashes' => $crashes,
        'count' => count($crashes)
    ]);
}

function getPopulation($config) {
    // Leer población actual desde el log de sobrevivientes
    $population = [];
    
    if (file_exists($config['log_file'])) {
        $lines = file($config['log_file'], FILE_IGNORE_NEW_LINES);
        
        foreach ($lines as $line) {
            if (preg_match('/Survivor (\d+): ([a-f0-9]+)/', $line, $matches)) {
                $population[] = [
                    'id' => $matches[1],
                    'shellcode' => $matches[2],
                    'length' => strlen($matches[2]) / 2
                ];
            }
        }
    }
    
    echo json_encode([
        'population' => $population,
        'count' => count($population)
    ]);
}

function getShellcodeDetails($config, $hash) {
    if (!$hash) {
        http_response_code(400);
        echo json_encode(['error' => 'Shellcode hash required']);
        return;
    }
    
    // Buscar en crashes
    $crashes = [];
    $dirs = [$config['crash_dir'], $config['critical_dir']];
    
    foreach ($dirs as $dir) {
        if (is_dir($dir)) {
            $files = glob($dir . "/*.json");
            
            foreach ($files as $file) {
                $data = json_decode(file_get_contents($file), true);
                if ($data && strpos($data['shellcode_hex'] ?? '', $hash) !== false) {
                    $crashes[] = $data;
                }
            }
        }
    }
    
    if (empty($crashes)) {
        http_response_code(404);
        echo json_encode(['error' => 'Shellcode not found']);
        return;
    }
    
    // Analizar el shellcode
    $shellcode = $crashes[0]['shellcode_hex'] ?? '';
    $analysis = analyzeShellcode($shellcode);
    
    echo json_encode([
        'shellcode' => $shellcode,
        'crashes' => $crashes,
        'analysis' => $analysis
    ]);
}

function getStats($config) {
    $stats = [
        'total_generations' => 0,
        'total_crashes' => 0,
        'critical_crashes' => 0,
        'avg_crash_rate' => 0,
        'most_common_crash' => '',
        'total_shellcodes' => 0,
        'avg_shellcode_length' => 0
    ];
    
    // Leer métricas
    if (file_exists($config['metrics_file'])) {
        $metrics = json_decode(file_get_contents($config['metrics_file']), true);
        
        if ($metrics) {
            $stats['total_generations'] = count($metrics['generations'] ?? []);
            $stats['avg_crash_rate'] = array_sum($metrics['crash_rates'] ?? []) / max(1, count($metrics['crash_rates'] ?? []));
            $stats['total_shellcodes'] = array_sum($metrics['shellcode_lengths'] ?? []);
            $stats['avg_shellcode_length'] = array_sum($metrics['shellcode_lengths'] ?? []) / max(1, count($metrics['shellcode_lengths'] ?? []));
            
            // Crash más común
            $crashTypes = [];
            foreach ($metrics['crash_types'] ?? [] as $genCrashes) {
                foreach ($genCrashes as $type => $count) {
                    $crashTypes[$type] = ($crashTypes[$type] ?? 0) + $count;
                }
            }
            
            if (!empty($crashTypes)) {
                arsort($crashTypes);
                $stats['most_common_crash'] = array_key_first($crashTypes);
            }
        }
    }
    
    // Contar crashes
    if (is_dir($config['crash_dir'])) {
        $stats['total_crashes'] = count(glob($config['crash_dir'] . "/*.json"));
    }
    
    if (is_dir($config['critical_dir'])) {
        $stats['critical_crashes'] = count(glob($config['critical_dir'] . "/*.json"));
    }
    
    echo json_encode($stats);
}

function calculateStats($metrics) {
    $stats = [
        'total_generations' => count($metrics['generations'] ?? []),
        'avg_crash_rate' => 0,
        'total_system_impacts' => 0,
        'crash_type_distribution' => [],
        'attack_distribution' => [],
        'mutation_distribution' => []
    ];
    
    if (!empty($metrics['crash_rates'])) {
        $stats['avg_crash_rate'] = array_sum($metrics['crash_rates']) / count($metrics['crash_rates']);
    }
    
    if (!empty($metrics['system_impacts'])) {
        $stats['total_system_impacts'] = array_sum($metrics['system_impacts']);
    }
    
    // Distribución de tipos de crash
    $crashTypes = [];
    foreach ($metrics['crash_types'] ?? [] as $genCrashes) {
        foreach ($genCrashes as $type => $count) {
            $crashTypes[$type] = ($crashTypes[$type] ?? 0) + $count;
        }
    }
    $stats['crash_type_distribution'] = $crashTypes;
    
    // Distribución de ataques
    $stats['attack_distribution'] = $metrics['attack_totals'] ?? [];
    
    // Distribución de mutaciones
    $stats['mutation_distribution'] = $metrics['mutation_totals'] ?? [];
    
    return $stats;
}

function analyzeShellcode($shellcode) {
    $analysis = [
        'length' => strlen($shellcode) / 2,
        'patterns' => [],
        'possible_instructions' => [],
        'entropy' => 0
    ];
    
    // Calcular entropía
    $bytes = str_split($shellcode, 2);
    $freq = array_count_values($bytes);
    $entropy = 0;
    $total = count($bytes);
    
    foreach ($freq as $count) {
        $p = $count / $total;
        $entropy -= $p * log($p, 2);
    }
    
    $analysis['entropy'] = round($entropy, 2);
    
    // Detectar patrones comunes
    $patterns = [
        'syscall' => ['0f05', 'cd80'],
        'nop' => ['90', '6690', '0f1f00'],
        'xor' => ['4831', '31'],
        'mov' => ['48c7', '4889', '48a1'],
        'push_pop' => ['50', '58', '51', '59']
    ];
    
    foreach ($patterns as $type => $patternList) {
        foreach ($patternList as $pattern) {
            $count = substr_count(strtolower($shellcode), strtolower($pattern));
            if ($count > 0) {
                $analysis['patterns'][$type] = ($analysis['patterns'][$type] ?? 0) + $count;
            }
        }
    }
    
    // Intentar interpretar algunas instrucciones
    $commonInstructions = [
        '4831c0' => 'xor rax, rax',
        '4831ff' => 'xor rdi, rdi',
        '48c7c0' => 'mov rax, imm32',
        '0f05' => 'syscall',
        '90' => 'nop',
        'c3' => 'ret'
    ];
    
    foreach ($commonInstructions as $hex => $instruction) {
        if (strpos(strtolower($shellcode), strtolower($hex)) !== false) {
            $analysis['possible_instructions'][] = $instruction;
        }
    }
    
    return $analysis;
}

function handleWebSocket() {
    // Implementación básica de WebSocket para actualizaciones en tiempo real
    // Esto requeriría una implementación más compleja con Ratchet o similar
    echo json_encode(['error' => 'WebSocket not implemented yet']);
}
?> 