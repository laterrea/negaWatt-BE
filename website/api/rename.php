<?php
/**
 * POST /api/rename.php   {code, group_id, token, name}
 *
 * A group that joined from a link is named "Group 3" by the server. In a room
 * with tables that is worth changing, so the play page offers a one-tap rename;
 * remotely nobody bothers and the ordinal is fine.
 */
declare(strict_types=1);
require_once __DIR__ . '/db.php';

ws_require_method(['POST']);
$pdo = ws_db();

$body = ws_body();
$session = ws_session_by(
    $pdo,
    isset($body['code']) ? ws_code($body['code']) : null,
    isset($body['slug']) ? ws_slug($body['slug']) : null
);
$code = $session['code'];
$groupId = (int) ($body['group_id'] ?? 0);
$token = ws_str($body['token'] ?? null, 96, 'token');
$name = ws_str($body['name'] ?? null, 80, 'name');

$stmt = $pdo->prepare('SELECT token_hash FROM ws_groups WHERE id = ? AND session_code = ?');
$stmt->execute([$groupId, $code]);
$hash = $stmt->fetchColumn();
if (!$hash) {
    ws_fail('unknown_group', 404);
}
if (!hash_equals((string) $hash, ws_hash((string) $token))) {
    ws_fail('bad_token', 403);
}

$stmt = $pdo->prepare('SELECT id FROM ws_groups WHERE session_code = ? AND name = ? AND id <> ?');
$stmt->execute([$code, $name, $groupId]);
if ($stmt->fetchColumn()) {
    ws_fail('name_taken', 409);
}

$stmt = $pdo->prepare('UPDATE ws_groups SET name = ?, updated_at = ? WHERE id = ?');
$stmt->execute([$name, ws_now(), $groupId]);

ws_json(['ok' => true, 'group_id' => $groupId, 'name' => $name]);
