<?php
/**
 * negaWatt-BE workshop — configuration template.
 *
 * The real file, config.php, is written on the server by
 * scripts/setup-negawatt-workshop-db.sh and is NOT in git. This sample only
 * documents the shape; the API refuses to start without a real config.php.
 *
 * /.htaccess denies web access to config.php and to *.sample.php.
 */
return [
    'db' => [
        'host'    => '127.0.0.1',
        'name'    => 'negawatt_ws',
        'user'    => 'negawatt_ws',
        'pass'    => 'CHANGE_ME',
        'charset' => 'utf8mb4',
    ],
    'base_url'  => 'https://negawatt.squoilin.eu',
];
