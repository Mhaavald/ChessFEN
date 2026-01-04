#!/bin/bash
set -e

# Create log directories
mkdir -p /var/log/supervisor
mkdir -p /var/run

# Start supervisord
exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf
