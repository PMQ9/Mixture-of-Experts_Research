import sys
import subprocess
from pathlib import Path

def main():
    script_path = Path(__file__).parent / 'src/Vision_Transformer_Pytorch/' / 'train_moe.py'
    script_path = script_path.resolve()
    args = sys.argv[1:]
    subprocess.run([sys.executable, str(script_path)] + args, check=True)

if __name__ == '__main__':
    main()