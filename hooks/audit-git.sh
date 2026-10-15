#!/usr/bin/env bash
# 审计 git 命令：记录到日志
echo "[$(date -Is)] git command: $*" >> .mcc/hooks/audit.log
exit 0
