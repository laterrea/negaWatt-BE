<?php
/**
 * POST /api/session.php   create a session   -> {code, admin_token, slug, …}
 * GET  /api/session.php?code=ABCD            -> the session and its groups
 * GET  /api/session.php?slug=namur-2026      -> same, addressed by its slug
 *
 * A slug is what makes a plain join link possible: the facilitator creates the
 * workshop once and shares …/workshop/?w=<slug>, and nobody has to type a code.
 */
declare(strict_types=1);
require_once __DIR__ . '/db.php';

ws_require_method(['GET', 'POST']);
$pdo = ws_db();

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $body = ws_body();
    $topic = ws_id($body['topic'] ?? null, 'topic');
    $label = ws_str($body['label'] ?? '', 160, 'label', false) ?? '';
    // Read once, then validate. Testing `$body['mode'] ?? 'group'` in the
    // condition and returning `$body['mode']` in the branch is the same bug that
    // inserted a null mode and produced a 500.
    $mode = $body['mode'] ?? 'group';
    if (!in_array($mode, ['group', 'solo'], true)) {
        $mode = 'group';
    }
    // A solo session is a private try-out: it never joins the topic-wide totals.
    $resultsPublic = !empty($body['results_public']) ? 1 : 0;
    $slug = ws_slug($body['slug'] ?? null, false);

    if ($slug !== null) {
        $stmt = $pdo->prepare('SELECT code FROM ws_sessions WHERE slug = ?');
        $stmt->execute([$slug]);
        if ($stmt->fetchColumn()) {
            ws_fail('slug_taken', 409, ['slug' => $slug]);
        }
    }

    $code = ws_new_code($pdo);
    $token = ws_token();
    $stmt = $pdo->prepare(
        'INSERT INTO ws_sessions
           (code, slug, topic, label, mode, results_public, reveal_step, admin_token_hash, created_at)
         VALUES (?, ?, ?, ?, ?, ?, -1, ?, ?)'
    );
    $stmt->execute([$code, $slug, $topic, $label, $mode, $resultsPublic,
                    ws_hash($token), ws_now()]);

    ws_json([
        'code' => $code, 'slug' => $slug, 'admin_token' => $token, 'topic' => $topic,
        'label' => $label, 'mode' => $mode, 'reveal_step' => -1,
    ], 201);
}

$session = ws_session_by(
    $pdo,
    isset($_GET['code']) ? ws_code($_GET['code']) : null,
    isset($_GET['slug']) ? ws_slug($_GET['slug']) : null
);
$code = $session['code'];

$stmt = $pdo->prepare(
    'SELECT g.id, g.name, g.updated_at, COUNT(a.id) AS answered
       FROM ws_groups g
       LEFT JOIN ws_answers a ON a.group_id = g.id
      WHERE g.session_code = ?
      GROUP BY g.id, g.name, g.updated_at
      ORDER BY g.id'
);
$stmt->execute([$code]);
$groups = [];
foreach ($stmt->fetchAll() as $row) {
    $groups[] = [
        'id' => (int) $row['id'],
        'name' => $row['name'],
        'answered' => (int) $row['answered'],
        'updated_at' => $row['updated_at'],
    ];
}

ws_json([
    'code' => $session['code'],
    'slug' => $session['slug'],
    'topic' => $session['topic'],
    'label' => $session['label'],
    'mode' => $session['mode'],
    'reveal_step' => (int) $session['reveal_step'],
    'results_public' => (bool) $session['results_public'],
    'closed' => $session['closed_at'] !== null,
    'created_at' => $session['created_at'],
    'groups' => $groups,
]);
