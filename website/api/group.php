<?php
/**
 * POST /api/group.php   {topic, name?, name_prefix?}   -> {group_id, token, name, topic}
 *
 * Start answering. There is nothing to join: a group belongs to a topic, and the
 * reveal screen picks up whichever groups answered in the window it is showing.
 *
 * `name` is optional — most participants just press Start. When it is empty the
 * server names the group after its rank in the day ("Groupe 3"); the prefix comes
 * from the client because only the client knows the language.
 */
declare(strict_types=1);
require_once __DIR__ . '/db.php';

ws_require_method(['POST']);
$pdo = ws_db();

$body = ws_body();
$topic = ws_id($body['topic'] ?? null, 'topic');
$name = ws_str($body['name'] ?? null, 80, 'name', false);
$prefix = ws_str($body['name_prefix'] ?? null, 40, 'name_prefix', false) ?? 'Group';

$token = ws_token();
$now = ws_now();

$stmt = $pdo->prepare(
    'INSERT INTO ws_groups (topic, name, token_hash, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?)'
);
$stmt->execute([$topic, $name ?? '', ws_hash($token), $now, $now]);
$groupId = (int) $pdo->lastInsertId();

if ($name === null) {
    // Rank within today, counted from the row itself, so two devices pressing
    // Start at the same instant cannot land on the same ordinal.
    $stmt = $pdo->prepare(
        'SELECT COUNT(*) FROM ws_groups
          WHERE topic = ? AND created_at >= ? AND id <= ?'
    );
    $stmt->execute([$topic, gmdate('Y-m-d 00:00:00'), $groupId]);
    $name = mb_substr($prefix . ' ' . (int) $stmt->fetchColumn(), 0, 80);
    $stmt = $pdo->prepare('UPDATE ws_groups SET name = ? WHERE id = ?');
    $stmt->execute([$name, $groupId]);
}

ws_json(['group_id' => $groupId, 'token' => $token, 'name' => $name,
         'topic' => $topic, 'created_at' => $now]);
