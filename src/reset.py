import shutil
import sys
from typing import List


try:

    if sys.argv[1].lower() == "hard":
        shutil.rmtree("..\\model_data")
        print("Success")
        sys.exit(0)
    elif sys.argv[1].lower() == "soft" :
        shutil.rmtree("model_data")
        print("Success")
        sys.exit(0)
    else:
        print("Invalid input")
        sys.exit(1)

except IndexError:
    print("No parameters given \n Input hard or soft as parameter")
    sys.exit(1)