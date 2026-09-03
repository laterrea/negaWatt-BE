<?php
/**
 * GET /api/results.php?topic=inland-mobility[&from=…][&to=…]
 *
 * Every group that started on this topic inside the window, with its answers.
 * `from`/`to` bound ws_groups.created_at and are UTC ('Y-m-d' or 'Y-m-d H:i:s');
 * a bare `to` date covers that whole day. Omit either bound for an open end —
 * omit both and you get every workshop ever run on the topic, which is how the
 * reveal screen summarises a series of sessions.
 *
 * No credential: the reveal screen is meant to be opened and projected without
 * anybody typing a key, and the data is slider values, not personal data.
 */
declare(strict_types=1);
require_once __DIR__ . '/db.php';

ws_require_method(['GET']);
$pdo = ws_db();

$topic = ws_id($_GET['topic'] ?? null, 'topic');
$from = ws_stamp($_GET['from'] ?? null, 'from');
$to = ws_stamp($_GET['to'] ?? null, 'to', true);

$sql = 'SELECT id, name, created_at, updated_at FROM ws_groups WHERE topic = ?';
$args = [$topic];
if ($from !== null) {
    $sql .= ' AND created_at >= ?';
    $args[] = $from;
}
if ($to !== null) {
    $sql .= ' AND created_at <= ?';
    $args[] = $to;
}
$sql .= ' ORDER BY id';

$stmt = $pdo->prepare($sql);
$stmt->execute($args);

$groups = [];
$groupIds = [];
foreach ($stmt->fetchAll() as $row) {
    $groups[] = ['id' => (int) $row['id'], 'name' => $row['name'],
                 'created_at' => $row['created_at'], 'updated_at' => $row['updated_at']];
    $groupIds[] = (int) $row['id'];
}

$answers = [];
if ($groupIds) {
    $marks = implode(',', array_fill(0, count($groupIds), '?'));
    $stmt = $pdo->prepare(
        "SELECT group_id, lever_id, value, confidence, condition_text, updated_at
           FROM ws_answers WHERE group_id IN ($marks)
          ORDER BY lever_id, group_id"
    );
    $stmt->execute($groupIds);
    foreach ($stmt->fetchAll() as $row) {
        $answers[] = [
            'group_id' => (int) $row['group_id'],
            'lever_id' => $row['lever_id'],
            'value' => (float) $row['value'],
            'confidence' => $row['confidence'] === null ? null : (int) $row['confidence'],
            'condition' => $row['condition_text'],
            'updated_at' => $row['updated_at'],
        ];
    }
}

ws_json([
    'topic' => $topic, 'from' => $from, 'to' => $to,
    'groups' => $groups, 'answers' => $answers, 'served_at' => ws_now(),
]);
