#!C:\Users\stanc\PycharmProjects\ExtraTrees\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'ensemble==0.0.dev1','console_scripts','ensemble-cli'
__requires__ = 'ensemble==0.0.dev1'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('ensemble==0.0.dev1', 'console_scripts', 'ensemble-cli')()
    )
