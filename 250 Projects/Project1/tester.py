import os
import subprocess
import difflib
import sys
from tqdm import tqdm

__author__ = "ahmetfirat"

try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm'])
    print("\n### Ignore previous lines. ###")
except:
    input('Try running "pip install tqdm" from your own console or edit the code and remove 5th line and all '
          'instances of "tqdm" word. By editing code you will be removing progress bars and no other functionality.')
    sys.exit()
try:
    subprocess.run(['javac', '*.java'])
except:
    input('Try running "javac *.java" from your own console in src. If it doesn\'t work google "How to Run Java '
          'Program in Command Prompt" and follow any tutorial that installs java jdk. If it works I have no idea what '
          'might be the problem.')
    sys.exit()
try:
    os.makedirs('res/outputs')
    os.makedirs('res/outputs2')
    os.makedirs('res/outputs3')
except:
    pass
print('\nIf progress bar doesn\'t move please close this script because your code probably entered endless loop and '
      'creating very big text output.')
print('Result file will be opened at the end of this process. #th set means #th folder and Input# means input#.txt in '
      'that folder.\nIf outputs totally match empty line printed under Input#. Your outputs are stored in res file.')
print('Be aware that if your code doesn\'t end up properly, result file may say overall unmatching input count is zero '
      'but if you swipe down you can still see unmatching outputs.\nAlso, you are not supposed to get errors from Java. '
      'If so you probably have some unhandled exceptions in your project.')
result = open('result.txt', 'w')
overall = 0
result.write(f"Total unmatching input count: {overall}")
result.write("\n First set")
for i in tqdm(range(1, 11)):
    with open(f'inputs/input{i}.txt') as inp:
        subprocess.run(['java', 'Project1', f"inputs/input{i}.txt", f'res/outputs/output{i}.txt'])
    with open(f'res/outputs/output{i}.txt') as output:
        with open(f'outputs/output{i}.txt') as real_output:
            result.write(f"\nInput{i}\n")
            diff = difflib.unified_diff(real_output.readlines(), output.readlines())
            count = 0
            for line in diff:
                result.write(line)
                count += 1
            if count > 0:
                result.write(f"\nUnmatching found\n")
                overall += 1
result.flush()

result.write("\n Second set")
for i in tqdm(range(1, 51)):
    with open(f'inputs2/input{i}.txt') as inp:
        subprocess.run(['java', 'Project1', f"inputs2/input{i}.txt", f'res/outputs2/output{i}.txt'])
    with open(f'res/outputs2/output{i}.txt') as output:
        with open(f'outputs2/output{i}.txt') as real_output:
            result.write(f"\nInput{i}\n")
            diff = difflib.unified_diff(real_output.readlines(), output.readlines())
            count = 0
            for line in diff:
                result.write(line)
                count += 1
            if count > 0:
                result.write(f"\nUnmatching found\n")
                overall += 1
result.flush()

result.write("\n Third set")
for i in tqdm(range(0, 100)):
    with open(f'inputs3/input{i}.txt') as inp:
        subprocess.run(['java', 'Project1', f"inputs3/input{i}.txt", f'res/outputs3/output{i}.txt'])
    with open(f'res/outputs3/output{i}.txt') as output:
        with open(f'outputs3/output{i}.txt') as real_output:
            result.write(f"\nInput{i}\n")
            diff = difflib.unified_diff(real_output.readlines(), output.readlines())
            count = 0
            for line in diff:
                result.write(line)
                count += 1
            if count > 0:
                result.write(f"\nUnmatching found\n")
                overall += 1

result.seek(0)
result.write(f"Total unmatching input count: {overall}")
if overall == 0:
    print("Good to go")
else:
    print(f"Total unmatching input count: {overall}")
result.close()
os.startfile('result.txt')
input("Press any key to exit")
