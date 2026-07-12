import os
import shutil

# Clean up unwanted -p directory
if os.path.exists("-p"):
    shutil.rmtree("-p", ignore_errors=True)
