<?php
/**
 * POST /api/group.php   {code|slug, name}                -> {group_id, token, …}
 * POST /api/group.php   {code|slug, auto_name, name_prefix} -> the next free
 *                                                             "Groupe 3"-style name
 *
 * `auto_name` is what lets a participant open a link and start answering without
 * typing anything: the server hands out the next free ordinal in the session. The
 * prefix comes from the client because only the client knows the language.
 *
 * With an explicit name, re-joining under a name that already exists returns that
 * same group with a freshly issued token. Two devices claiming one group name is a
 * facilitator slip, not an attack, and the friendly behaviour (last device wins,
 * answers stay with the group) beats making people invent a new name mid-workshop.
 * See docs/workshop_module.md.
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

$auto = !empty($body['auto_name']);
$name = $auto
    ? null
    : ws_str($body['name'] ?? null, 80, 'name');

if ($session['closed_at'] !== null) {
    ws_fail('session_closed', 409);
}

$token = ws_token();
$now = ws_now();

if ($auto) {
    // Take the next free ordinal. Racing devices simply land on different
    // numbers, because the unique index rejects a collision and we try again.
    $prefix = ws_str($body['name_prefix'] ?? null, 40, 'name_prefix', false) ?? 'Group';
    $stmt = $pdo->prepare(
        'INSERT INTO ws_groups (session_code, name, token_hash, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?)'
    );
    $countStmt = $pdo->prepare('SELECT COUNT(*) FROM ws_groups WHERE session_code = ?');
    $countStmt->execute([$code]);
    $start = (int) $countStmt->fetchColumn() + 1;
    for ($n = $start; $n < $start + 200; $n++) {
        $candidate = mb_substr($prefix . ' ' . $n, 0, 80);
        try {
            $stmt->execute([$code, $candidate, ws_hash($token), $now, $now]);
            ws_json([
                'group_id' => (int) $pdo->lastInsertId(), 'token' => $token,
                'name' => $candidate, 'ordinal' => $n,
                'topic' => $session['topic'], 'code' => $code,
                'slug' => $session['slug'], 'rejoined' => false,
            ]);
        } catch (PDOException $e) {
            if ($e->getCode() !== '23000') {          // not a duplicate: real failure
                throw $e;
            }
        }
    }
    ws_fail('too_many_groups', 503);
}

$stmt = $pdo->prepare('SELECT id FROM ws_groups WHERE session_code = ? AND name = ?');
$stmt->execute([$code, $name]);
$existing = $stmt->fetchColumn();

if ($existing) {
    $stmt = $pdo->prepare('UPDATE ws_groups SET token_hash = ?, updated_at = ? WHERE id = ?');
    $stmt->execute([ws_hash($token), $now, (int) $existing]);
    $groupId = (int) $existing;
    $rejoined = true;
} else {
    $stmt = $pdo->prepare(
        'INSERT INTO ws_groups (session_code, name, token_hash, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?)'
    );
    $stmt->execute([$code, $name, ws_hash($token), $now, $now]);
    $groupId = (int) $pdo->lastInsertId();
    $rejoined = false;
}

ws_json([
    'group_id' => $groupId, 'token' => $token, 'name' => $name,
    'topic' => $session['topic'], 'code' => $code, 'slug' => $session['slug'],
    'rejoined' => $rejoined,
]);
