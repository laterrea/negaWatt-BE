<?php
/**
 * GET /api/results.php?code=ABCD[&admin_token=…]   one session
 * GET /api/results.php?topic=inland-mobility&admin_key=…   every OPEN session
 *
 * The topic scope is what makes a fully remote workshop work: each participant
 * can create their own session and the reveal screen still aggregates them.
 *
 * Gating: a session's results need its admin_token unless the session was created
 * with results_public. The topic scope needs the facilitator key from config.php.
 * Solo sessions are excluded from the topic scope on purpose.
 */
declare(strict_types=1);
require_once __DIR__ . '/db.php';

ws_require_method(['GET']);
$pdo = ws_db();

$topic = isset($_GET['topic']) ? ws_id($_GET['topic'], 'topic') : null;
$code = isset($_GET['code']) ? ws_code($_GET['code']) : null;

if ($code === null && $topic === null) {
    ws_fail('missing_field', 400, ['field' => 'code or topic']);
}

$codes = [];
$scope = 'session';
$sessions = [];

if ($code !== null) {
    $session = ws_session($pdo, $code);
    $adminToken = isset($_GET['admin_token']) ? (string) $_GET['admin_token'] : null;
    if (!$session['results_public'] && !ws_is_admin($session, $adminToken)) {
        ws_fail('forbidden', 403, ['detail' => 'admin_token required for this session']);
    }
    $codes = [$code];
    $topic = $session['topic'];
    $sessions[] = ['code' => $code, 'label' => $session['label'],
                   'reveal_step' => (int) $session['reveal_step'],
                   'closed' => $session['closed_at'] !== null];
} else {
    $scope = 'topic';
    $key = isset($_GET['admin_key']) ? (string) $_GET['admin_key'] : '';
    $expected = (string) (ws_config()['admin_key'] ?? '');
    if ($expected === '' || !hash_equals($expected, $key)) {
        ws_fail('forbidden', 403, ['detail' => 'admin_key required for the topic scope']);
    }
    $stmt = $pdo->prepare(
        "SELECT code, label, reveal_step, closed_at FROM ws_sessions
          WHERE topic = ? AND closed_at IS NULL AND mode <> 'solo'
          ORDER BY created_at"
    );
    $stmt->execute([$topic]);
    foreach ($stmt->fetchAll() as $row) {
        $codes[] = $row['code'];
        $sessions[] = ['code' => $row['code'], 'label' => $row['label'],
                       'reveal_step' => (int) $row['reveal_step'],
                       'closed' => $row['closed_at'] !== null];
    }
}

$groups = [];
$answers = [];

if ($codes) {
    $marks = implode(',', array_fill(0, count($codes), '?'));
    $stmt = $pdo->prepare(
        "SELECT id, session_code, name FROM ws_groups
          WHERE session_code IN ($marks) ORDER BY session_code, id"
    );
    $stmt->execute($codes);
    $groupIds = [];
    foreach ($stmt->fetchAll() as $row) {
        $groups[] = ['id' => (int) $row['id'], 'session' => $row['session_code'],
                     'name' => $row['name']];
        $groupIds[] = (int) $row['id'];
    }

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
}

ws_json([
    'scope' => $scope, 'topic' => $topic,
    'sessions' => $sessions, 'groups' => $groups, 'answers' => $answers,
    'served_at' => ws_now(),
]);
