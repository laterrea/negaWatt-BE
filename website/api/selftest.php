<?php
/**
 * GET /api/selftest.php — is the API wired up correctly?
 *
 * Reports the PHP version, whether config.php was found, whether the database
 * answers, and which tables exist with how many rows. Safe to leave in place:
 * it reveals no credentials and no participant data.
 */
declare(strict_types=1);
require_once __DIR__ . '/db.php';

ws_require_method(['GET']);

$out = [
    'ok' => false,
    'php' => PHP_VERSION,
    'configured' => is_file(__DIR__ . '/config.php'),
    'pdo_mysql' => in_array('mysql', PDO::getAvailableDrivers(), true),
];

$pdo = ws_db();
$out['database'] = 'reachable';

$expected = ['ws_groups', 'ws_answers', 'ws_answer_log'];
$out['tables'] = [];
$missing = [];
foreach ($expected as $table) {
    try {
        $count = (int) $pdo->query("SELECT COUNT(*) FROM `$table`")->fetchColumn();
        $out['tables'][$table] = $count;
    } catch (Throwable $e) {
        $missing[] = $table;
    }
}
if ($missing) {
    $out['missing_tables'] = $missing;
    $out['detail'] = 'apply website/api/schema.sql';
    ws_json($out, 503);
}

$out['ok'] = true;
ws_json($out);
