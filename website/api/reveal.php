<?php
/**
 * POST /api/reveal.php   {code, admin_token, step}
 *
 * Moves the reveal pointer so that several screens (the projector, the
 * facilitator's phone) stay on the same lever. -1 means nothing revealed yet.
 * Passing close:true ends the session, which also removes it from the
 * topic-wide aggregation.
 */
declare(strict_types=1);
require_once __DIR__ . '/db.php';

ws_require_method(['POST']);
$pdo = ws_db();

$body = ws_body();
$code = ws_code($body['code'] ?? null);
$session = ws_session($pdo, $code);

if (!ws_is_admin($session, isset($body['admin_token']) ? (string) $body['admin_token'] : null)) {
    ws_fail('forbidden', 403, ['detail' => 'admin_token required']);
}

$step = (int) ($body['step'] ?? -1);
if ($step < -1 || $step > 999) {
    ws_fail('invalid_field', 400, ['field' => 'step']);
}

if (!empty($body['close'])) {
    $stmt = $pdo->prepare('UPDATE ws_sessions SET reveal_step = ?, closed_at = ? WHERE code = ?');
    $stmt->execute([$step, ws_now(), $code]);
} else {
    $stmt = $pdo->prepare('UPDATE ws_sessions SET reveal_step = ? WHERE code = ?');
    $stmt->execute([$step, $code]);
}

ws_json(['ok' => true, 'code' => $code, 'reveal_step' => $step,
         'closed' => !empty($body['close'])]);
