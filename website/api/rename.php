<?php
/**
 * POST /api/rename.php   {group_id, token, name}
 *
 * A group that pressed Start without typing anything is named "Group 3" by the
 * server. In a room with tables that is worth changing, so the play page offers a
 * one-tap rename; remotely nobody bothers and the ordinal is fine.
 *
 * Duplicate names are allowed: two tables calling themselves "Table 3" is a
 * cosmetic problem for one workshop, and refusing the rename mid-session is worse.
 * The reveal screen disambiguates identical labels when it draws them.
 */
declare(strict_types=1);
require_once __DIR__ . '/db.php';

ws_require_method(['POST']);
$pdo = ws_db();

$body = ws_body();
$group = ws_group($pdo, $body['group_id'] ?? null, $body['token'] ?? null);
$name = ws_str($body['name'] ?? null, 80, 'name');

$stmt = $pdo->prepare('UPDATE ws_groups SET name = ?, updated_at = ? WHERE id = ?');
$stmt->execute([$name, ws_now(), (int) $group['id']]);

ws_json(['ok' => true, 'group_id' => (int) $group['id'], 'name' => $name]);
