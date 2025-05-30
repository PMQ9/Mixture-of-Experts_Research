@echo off
echo Stopping GitLab Runner...
cd /d "C:\GitLab-Runner"
gitlab-runner.exe stop
gitlab-runner.exe status
timeout /t 3
exit