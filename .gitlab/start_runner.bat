@echo off
echo Starting GitLab Runner...
cd /d "C:\GitLab-Runner"
gitlab-runner.exe start
gitlab-runner.exe status
timeout /t 3
exit