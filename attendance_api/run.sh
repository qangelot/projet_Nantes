#!/usr/bin/env bash
export IS_DEBUG=${DEBUG:-false}
exec uvicorn app.main:app --host 0.0.0.0:5000 --access-log
