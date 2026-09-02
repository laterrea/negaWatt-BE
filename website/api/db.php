<?php
/**
 * negaWatt-BE workshop — shared bootstrap: config, PDO handle, request helpers.
 *
 * Follows the same shape as the smartcampus site on this host
 * (includes/db.php → sc_config/sc_db/sc_json), so there is one pattern to learn.
 *
 *   require_once __DIR__ . '/db.php';
 *   $pdo = ws_db();
 *   ws_json(['ok' => true]);
 */
declare(strict_types=1);

const WS_CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';  // no O/0/I/1/L
const WS_CODE_LENGTH = 4;
const WS_MAX_BODY = 8192;

/**
 * Last line of defence: turn any uncaught error into a JSON 500 instead of the
 * blank body Apache would otherwise return. The detail goes to the error log,
 * never to the client.
 */
set_exception_handler(static function (Throwable $e): void {
    error_log('[nw-workshop] uncaught: ' . $e->getMessage() . ' @ '
              . $e->getFile() . ':' . $e->getLine());
    if (!headers_sent()) {
        ws_json(['error' => 'server_error'], 500);
    }
});

set_error_handler(static function (int $severity, string $message, string $file, int $line): bool {
    // Promote warnings and notices so a typo cannot silently produce a null.
    if (!(error_reporting() & $severity)) {
        return false;
    }
    throw new ErrorException($message, 0, $severity, $file, $line);
});

function ws_config(): array
{
    static $cfg = null;
    if ($cfg !== null) {
        return $cfg;
    }
    $local = __DIR__ . '/config.php';
    if (!is_file($local)) {
        ws_json(['error' => 'not_configured',
                 'detail' => 'config.php is missing; run setup-negawatt-workshop-db.sh'], 503);
    }
    $cfg = require $local;
    return $cfg;
}

function ws_db(): PDO
{
    static $pdo = null;
    if ($pdo !== null) {
        return $pdo;
    }
    $c = ws_config()['db'];
    $dsn = sprintf('mysql:host=%s;dbname=%s;charset=%s', $c['host'], $c['name'], $c['charset']);
    try {
        $pdo = new PDO($dsn, $c['user'], $c['pass'], [
            PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
            PDO::ATTR_EMULATE_PREPARES   => false,
        ]);
    } catch (PDOException $e) {
        // Never leak credentials or the DSN to the client.
        error_log('[nw-workshop] database unavailable: ' . $e->getMessage());
        ws_json(['error' => 'database_unavailable'], 503);
    }
    return $pdo;
}

/** Send JSON and stop. */
function ws_json($data, int $status = 200): void
{
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-store');
    // The workshop pages may be served from another origin (a laptop, file://,
    // the facilitator's screen), and nothing here is personal data.
    header('Access-Control-Allow-Origin: *');
    header('Access-Control-Allow-Headers: Content-Type');
    header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
    echo json_encode($data, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    exit;
}

function ws_fail(string $error, int $status = 400, array $extra = []): void
{
    ws_json(array_merge(['error' => $error], $extra), $status);
}

/** Answer a CORS preflight and stop. */
function ws_handle_preflight(): void
{
    if (($_SERVER['REQUEST_METHOD'] ?? 'GET') === 'OPTIONS') {
        ws_json(['ok' => true]);
    }
}

/** Require one of the given HTTP methods. */
function ws_require_method(array $allowed): void
{
    ws_handle_preflight();
    $method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
    if (!in_array($method, $allowed, true)) {
        ws_fail('method_not_allowed', 405, ['allowed' => $allowed]);
    }
}

/** Decode the JSON request body. */
function ws_body(): array
{
    $raw = file_get_contents('php://input', false, null, 0, WS_MAX_BODY + 1);
    if ($raw === false || $raw === '') {
        return [];
    }
    if (strlen($raw) > WS_MAX_BODY) {
        ws_fail('body_too_large', 413);
    }
    $data = json_decode($raw, true);
    if (!is_array($data)) {
        ws_fail('invalid_json');
    }
    return $data;
}

/* ------------------------------------------------------------------ validation */

function ws_str($value, int $max, string $field, bool $required = true): ?string
{
    if ($value === null || $value === '') {
        if ($required) {
            ws_fail('missing_field', 400, ['field' => $field]);
        }
        return null;
    }
    if (!is_string($value)) {
        ws_fail('invalid_field', 400, ['field' => $field]);
    }
    $value = trim($value);
    if (mb_strlen($value) > $max) {
        $value = mb_substr($value, 0, $max);
    }
    if ($value === '') {
        if ($required) {
            ws_fail('missing_field', 400, ['field' => $field]);
        }
        return null;
    }
    return $value;
}

function ws_id($value, string $field): string
{
    $value = (string) ($value ?? '');
    if (!preg_match('/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/', $value)) {
        ws_fail('invalid_field', 400, ['field' => $field]);
    }
    return $value;
}

function ws_code($value): string
{
    $value = strtoupper(trim((string) ($value ?? '')));
    if (!preg_match('/^[' . WS_CODE_ALPHABET . ']{' . WS_CODE_LENGTH . ',8}$/', $value)) {
        ws_fail('invalid_code', 400);
    }
    return $value;
}

/** A workshop slug: lowercase, hyphenated, human-readable, unique. */
function ws_slug($value, bool $required = true): ?string
{
    $value = strtolower(trim((string) ($value ?? '')));
    if ($value === '') {
        if ($required) {
            ws_fail('missing_field', 400, ['field' => 'slug']);
        }
        return null;
    }
    if (!preg_match('/^[a-z0-9][a-z0-9-]{1,63}$/', $value)) {
        ws_fail('invalid_slug', 400);
    }
    return $value;
}

function ws_number($value, string $field): float
{
    if (!is_int($value) && !is_float($value) &&
        !(is_string($value) && is_numeric($value))) {
        ws_fail('invalid_field', 400, ['field' => $field]);
    }
    $number = (float) $value;
    if (!is_finite($number)) {
        ws_fail('invalid_field', 400, ['field' => $field]);
    }
    return $number;
}

/* ---------------------------------------------------------------- credentials */

function ws_token(): string
{
    return bin2hex(random_bytes(24));
}

function ws_hash(string $token): string
{
    return hash('sha256', $token);
}

function ws_new_code(PDO $pdo): string
{
    $alphabet = WS_CODE_ALPHABET;
    $n = strlen($alphabet);
    for ($attempt = 0; $attempt < 40; $attempt++) {
        $code = '';
        for ($i = 0; $i < WS_CODE_LENGTH; $i++) {
            $code .= $alphabet[random_int(0, $n - 1)];
        }
        $stmt = $pdo->prepare('SELECT 1 FROM ws_sessions WHERE code = ?');
        $stmt->execute([$code]);
        if (!$stmt->fetchColumn()) {
            return $code;
        }
    }
    ws_fail('code_space_exhausted', 503);
}

function ws_now(): string
{
    return gmdate('Y-m-d H:i:s');
}

/** Load a session by code, or stop with 404. */
function ws_session(PDO $pdo, string $code): array
{
    $stmt = $pdo->prepare('SELECT * FROM ws_sessions WHERE code = ?');
    $stmt->execute([$code]);
    $row = $stmt->fetch();
    if (!$row) {
        ws_fail('unknown_session', 404);
    }
    return $row;
}

/**
 * Load a session from whichever identifier the caller has.
 *
 * Participants arrive from a link carrying the slug and never see the code;
 * someone reading a projector types the code. Both must work everywhere.
 */
function ws_session_by(PDO $pdo, ?string $code, ?string $slug): array
{
    if ($slug !== null && $slug !== '') {
        $stmt = $pdo->prepare('SELECT * FROM ws_sessions WHERE slug = ?');
        $stmt->execute([$slug]);
        $row = $stmt->fetch();
        if (!$row) {
            ws_fail('unknown_session', 404);
        }
        return $row;
    }
    if ($code === null || $code === '') {
        ws_fail('missing_field', 400, ['field' => 'code or slug']);
    }
    return ws_session($pdo, $code);
}

function ws_is_admin(array $session, ?string $token): bool
{
    return $token !== null && $token !== ''
        && hash_equals($session['admin_token_hash'], ws_hash($token));
}
