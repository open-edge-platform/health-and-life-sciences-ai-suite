# Uninstall

This guide removes a local Ultrasound Ovarian Segmentation deployment and restores the clone directory to its pre-run state.

## 1. Stop running processes

- Stop an active web dashboard run with the **Stop** button.
- Close the OpenCV window or press `q` or `Esc`.
- Stop any remaining PowerShell process for `run.ps1` or `run_webapp.ps1`.

## 2. Remove generated application state

From the `ultrasound-ovarian-segmentation` directory, remove generated model and runtime assets when you do not need them:

```powershell
Remove-Item -Recurse -Force .\venv
Remove-Item -Recurse -Force .\models
Remove-Item -Recurse -Force .\webapp\ui\node_modules
Remove-Item -Recurse -Force .\webapp\ui\dist
Remove-Item -Recurse -Force .\monitoring\tools
```

Do not remove `models` or `monitoring\tools` if another local application or benchmark shares those directories.

## 3. Remove downloaded or generated data

Remove only the data you created locally and no longer need:

```powershell
Remove-Item -Recurse -Force .\data
Remove-Item -Recurse -Force .\cache
Remove-Item -Recurse -Force .\output
```

If the dataset or source files came from shared storage, remove the local copy only after confirming it is not used by another workflow.

## 4. Remove the repository

After the required data and local copies are backed up or no longer needed:

```powershell
cd ..
Remove-Item -Recurse -Force .\health-and-life-sciences-ai-suite
```

This removes the application source and all remaining clone-local files.
