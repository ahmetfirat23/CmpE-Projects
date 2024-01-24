#Import os Library
import os
import time

# Fork a child process
processid = os.fork()
print(processid)

# processid > 0 represents the parent process
if processid > 0 :
  
  print("\nParent Process:")
  print("Process ID:", os.getpid())
  print("Child's process ID:", processid)
  time.sleep(1)
  print("Parent is done")

# processid = 0 represents the created child process
else :  
  print("\nChild Process:")
  print("Process ID:", os.getpid())
  print("Parent's process ID:", os.getppid())
  processid = os.fork()
  print(processid)
  time.sleep(1)
  print("Child is done")