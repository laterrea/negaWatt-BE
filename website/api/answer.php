<?php
/**
 * POST /api/answer.php
 *   {code, group_id, token, lever_id, value, confidence?, condition?}
 *
 * Upserts the group's current answer and appends a row to the trace log, so the
 * history of a group's thinking survives the workshop.
 */
declare(strict_types=1);
require_once __DIR__ . '/db.php';

ws_require_method(['POST']);
$pdo = ws_db();

$body = ws_body();
$code = ws_code($body['code'] ?? null);
$groupId = (int) ($body['group_id'] ?? 0);
$token = ws_str($body['token'] ?? null, 96, 'token');
$leverId = ws_id($body['lever_id'] ?? null, 'lever_id');
$value = ws_number($body['value'] ?? null, 'value');

$confidence = null;
if (isset($body['confidence']) && $body['confidence'] !== null) {
    $confidence = (int) $body['confidence'];
    if ($confidence < 1 || $confidence > 3) {
        ws_fail('invalid_field', 400, ['field' => 'confidence']);
    }
}
$condition = ws_str($body['condition'] ?? null, 280, 'condition', false);

$session = ws_session($pdo, $code);
if ($session['closed_at'] !== null) {
    ws_fail('session_closed', 409);
}

$stmt = $pdo->prepare('SELECT token_hash FROM ws_groups WHERE id = ? AND session_code = ?');
$stmt->execute([$groupId, $code]);
$hash = $stmt->fetchColumn();
if (!$hash) {
    ws_fail('unknown_group', 404);
}
if (!hash_equals((string) $hash, ws_hash((string) $token))) {
    ws_fail('bad_token', 403);
}

$now = ws_now();
$pdo->beginTransaction();
try {
    $stmt = $pdo->prepare(
        'INSERT INTO ws_answers (group_id, lever_id, value, confidence, condition_text, updated_at)
         VALUES (?, ?, ?, ?, ?, ?)
         ON DUPLICATE KEY UPDATE
           value = VALUES(value), confidence = VALUES(confidence),
           condition_text = VALUES(condition_text), updated_at = VALUES(updated_at)'
    );
    $stmt->execute([$groupId, $leverId, $value, $confidence, $condition, $now]);

    $stmt = $pdo->prepare(
        'INSERT INTO ws_answer_log (group_id, lever_id, value, confidence, created_at)
         VALUES (?, ?, ?, ?, ?)'
    );
    $stmt->execute([$groupId, $leverId, $value, $confidence, $now]);

    $stmt = $pdo->prepare('UPDATE ws_groups SET updated_at = ? WHERE id = ?');
    $stmt->execute([$now, $groupId]);

    $pdo->commit();
} catch (Throwable $e) {
    $pdo->rollBack();
    error_log('[nw-workshop] answer failed: ' . $e->getMessage());
    ws_fail('write_failed', 500);
}

ws_json(['ok' => true, 'lever_id' => $leverId, 'updated_at' => $now]);
